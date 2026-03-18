from typing import Dict 
import torch
import torch.fft
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from torch.fft import fft2, ifft2, fftshift, ifftshift
class ResidualPredictor:
    def __init__(self, order=3, history_size=5, N=0.6, device="cuda", dtype=torch.float32):
        self.threshold = N
        self.order = order
        self.history_size = history_size
        self.device = device
        self.dtype = dtype
        # history residual storage
        self.residual_history = deque(maxlen=history_size)
        
        # adaptive hyperparameter
        self.current_order = order
        self.stability_factor = 1.0
        self.step_count = 0
        self.total_timesteps = 20
        # state
        self.initialized = False
        self.last_input = None
  

    def reset(self):
        self.residual_history.clear()
        self.current_order = self.order
        self.stability_factor = 1.0
        self.step_count = 0
        self.initialized = False
        self.last_input = None
    
    def _calculate_residual(self, hidden_in, hidden_out):
        return hidden_out - hidden_in
    def cosine_similarity(self, a, b):
        a = a.view(-1)
        b = b.view(-1)
        if torch.norm(a) == 0 or torch.norm(b) == 0:
            return 0.0
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    def _adaptive_order_selection(self):
        if len(self.residual_history) < 3:
            return min(2, self.order)

        similarities = []
        for i in range(1, min(3, len(self.residual_history))):
            sim = self.cosine_similarity(
                self.residual_history[-i],
                self.residual_history[-i - 1]
            )
            similarities.append(sim)

        avg_similarity = np.mean(similarities)

        # cosine similarity
        if avg_similarity > 0.98:
            return min(self.order, 4)
        elif avg_similarity > 0.90:
            return min(self.order, 3)
        else:
            return 2    
    def _weighted_extrapolation(self):
        weights = [0.4, 0.3, 0.2, 0.1]
        
        # ensure enough history
        k = min(len(self.residual_history), self.current_order)
        if k == 0:
            return None
        
        # weighted average
        prediction = torch.zeros_like(self.residual_history[-1])
        total_weight = sum(weights[:k])
        
        for i in range(k):
            weight = weights[i] / total_weight
            prediction += weight * self.residual_history[-1-i]
        
        # apply stability factor
        return prediction * self.stability_factor
    
    def _pade_prediction(self):
        if len(self.residual_history) < self.current_order + 1:
            return self._weighted_extrapolation()
        
        try:
            # adaptive coefficients - based on curvature
            delta = torch.abs(self.residual_history[-1] - self.residual_history[-2])
            avg_magnitude = 0.5 * (torch.abs(self.residual_history[-1]) + 
                                torch.abs(self.residual_history[-2]))
            
            # stability factor - more conservative when change is large
            stability_factor = torch.exp(-5.0 * delta / (avg_magnitude + 1e-7))
            
            # adaptive coefficients
            b0 = 2 * stability_factor
            b1 = 1 * stability_factor
            # curvature estimation, get sign of second difference
            curvature_sign = self.curvetest()

            # determine sign of a1 based on curvature
            if curvature_sign < 0:  
                a1 = 0.1 * stability_factor  
            else:  
                a1 = -0.1 * stability_factor  
            
            # numerator part: b0*R_n + b1*R_{n-1}
            numerator = b0 * self.residual_history[-1] - b1 * self.residual_history[-2]
            
            # denominator part: 1 + a1*R_{n-2} (using earlier history point)
            # denominator = 1.0 + a1 * self.residual_history[-3] if len(self.residual_history) >= 3 else 1.0
            denominator = 1.0 + a1 * curvature_sign
            # enhanced stability protection
            abs_denom = torch.abs(denominator)
            
            # use approximate value when denominator is too small
            safe_denom = torch.where(
                abs_denom < 1e-5,
                1.0 + a1 * self.residual_history[-1] if len(self.residual_history) >= 2 else 1.0,
                denominator
            )
            
            denom_sign = torch.sign(safe_denom)
            safe_denom = denom_sign * torch.maximum(1e-5 * torch.ones_like(abs_denom), abs_denom)
            
            # calculate prediction value
            result = numerator / safe_denom
            
            # historical consistency constraint
            recent_avg = 0.6 * self.residual_history[-1] + 0.4 * self.residual_history[-2]
            
            # calculate acceptable deviation range
            max_deviation = 0.5 * torch.abs(self.residual_history[-1] - self.residual_history[-2])
            max_deviation = torch.maximum(max_deviation, 0.1 * avg_magnitude)
            
            deviation = result - recent_avg
            deviation = torch.clamp(deviation, -max_deviation, max_deviation)
            result = recent_avg + deviation

            # blend historical average and current prediction
            blend_factor = 0.7 * stability_factor + 0.3
            result = blend_factor * result + (1 - blend_factor) * recent_avg
            
            if torch.isnan(result).any() or torch.isinf(result).any():
                try:
                    result = self.residual_history[-1] + 0.5 * (self.residual_history[-1] - self.residual_history[-2])
                    if torch.isnan(result).any() or torch.isinf(result).any():
                        raise ValueError("Fallback failed")
                except:
                    print("[Warning] Final fallback to weighted extrapolation")
                    return self._weighted_extrapolation()
            
            return result

        except Exception as e:
            print(f"[Exception] Pade prediction failed: {e}")
            return self._weighted_extrapolation()

    def update_history(self, hidden_in, hidden_out):
        """Update history residual records"""
        residual = self._calculate_residual(hidden_in, hidden_out)
        self.residual_history.append(residual.detach().clone().to(self.device))
        self.last_input = hidden_in.detach().clone().to(self.device)
        self.step_count += 1
        
        # Update adaptive parameters
        if len(self.residual_history) >= 3:
            self.current_order = self._adaptive_order_selection()
            
            # Update stability factor (0.8-1.2)
            last_change = torch.norm(self.residual_history[-1] - self.residual_history[-2]).item()
            self.stability_factor = max(0.8, min(1.2, 1.0 / (1.0 + 10 * last_change)))
        
        # Check initialization status
        if not self.initialized and len(self.residual_history) >= min(3, self.order):
            self.initialized = True

    def _high_noise_prediction(self):
        """Early stage: simplified prediction"""
        return self.residual_history[-1] * 0.9 + self.residual_history[-2] * 0.1

    def _detail_enhanced_prediction(self):
        """Later stage: detail-enhanced prediction"""
        # calculate residual gradient to enhance details
        grad = self.residual_history[-1] - self.residual_history[-2]
        pade_pred = self._pade_prediction()
        return pade_pred + 0.3 * grad
    
    def predict_residual(self, timestep):
        """Time-step aware residual prediction"""
        if not self.initialized or len(self.residual_history) == 0:
            return None
        
        # Select prediction strategy based on timestep
        if timestep > 0.8 * self.total_timesteps:  # early high-noise stage
            return self._detail_enhanced_prediction()
        elif timestep > 0.3 * self.total_timesteps:  # middle stage
            return self._pade_prediction()
        else:  # later detail-enhanced stage
            return self._high_noise_prediction()

    def predict_output(self, current_input, cur_step):
        """Predict full output state"""
        residual_pred = self.predict_residual(cur_step)
        if residual_pred is None or self.last_input is None:
            return None
        
        # Calculate input change
        input_delta = current_input - self.last_input
        
        # Adjust predicted residual (considering input change)
        adjusted_residual = residual_pred + 0.2 * input_delta
        output = current_input + adjusted_residual
        return output    

    def curvetest(self):
        """Curvature estimation"""
        if len(self.residual_history) < 5:
            return True

        def bend(r0, r1, r2, eps=1e-12):
            v1 = (r1 - r0).flatten(); v2 = (r2 - r1).flatten()
            n1 = torch.norm(v1) + eps; n2 = torch.norm(v2) + eps
            u1, u2 = v1/n1, v2/n2
            return 0.5 * torch.norm(u1 - u2) # ∈ [0,1]

        r0 = self.residual_history[-3]
        r1 = self.residual_history[-2]
        r2 = self.residual_history[-1]
        score = bend(r0, r1, r2)

        return bool(score < self.threshold)