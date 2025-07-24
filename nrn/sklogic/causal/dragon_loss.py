import torch
import torch.nn as nn
import torch.nn.functional as F

class DragonLoss(nn.Module):
    def __init__(self, alpha=0.5, lambda_reg=0.5, lambda_tar=0.0, use_brier=False, 
                 clip_propensity=None, balance_classes=False, epsilon_init=0.01):
        """
        Enhanced DragonNet Loss with multiple regularization options.
        
        Parameters:
        - alpha: weight on propensity loss
        - lambda_reg: weight on PEHE regularization (treatment effect variance)
        - lambda_tar: weight on targeted regularization (IPW-based)
        - use_brier: if True, use Brier score instead of cross-entropy for propensity
        - clip_propensity: if True, clip propensity scores for numerical stability
        - balance_classes: if True, apply class balancing to propensity loss
        - epsilon_init: initial value for learnable epsilon in targeted reg
        """
        super(DragonLoss, self).__init__()
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.lambda_tar = lambda_tar
        self.use_brier = use_brier
        self.clip_propensity = clip_propensity
        self.balance_classes = balance_classes
        self.eps = 1e-6

    
        
        # Learnable epsilon for targeted regularization
        if lambda_tar > 0:
            self.epsilon = nn.Parameter(torch.tensor(epsilon_init, dtype=torch.float32))
        else:
            self.register_parameter('epsilon', None)

    def forward(self, y0_hat, y1_hat, g_logits, t, y, return_components=False):
        """
        Inputs:
        - y0_hat: predicted outcomes under control (batch,)
        - y1_hat: predicted outcomes under treatment (batch,)
        - g_logits: logits for propensity scores (batch,) - will be sigmoid'd
        - t: binary treatment indicators (batch,)
        - y: observed outcomes (batch,)
        - return_components: if True, return loss components for monitoring
        """

        # Ensure proper shapes and types
        batch_size = y0_hat.shape[0]
        t = t.float()
        y = y.float()
        y0_hat = y0_hat
        y1_hat = y1_hat
        g_logits = g_logits
        
        # Convert logits to probabilities
        g = g_logits #torch.sigmoid(g_logits)
        #print(g)
        
        # Optional propensity clipping for numerical stability
        #if self.clip_propensity:
        #g = torch.clamp(g, self.eps, 1 - self.eps)
        
        # 1. Factual outcome loss (MSE on observed outcomes)
        factual_loss = t * (y1_hat - y) ** 2 + (1 - t) * (y0_hat - y) ** 2
        factual_loss = factual_loss.mean()
        
        # 2. Propensity loss
        if self.use_brier:
            # Brier score: MSE between probabilities and binary outcomes
            prop_loss = F.mse_loss(g, t)
        else:
            # Binary cross-entropy
            prop_loss = F.binary_cross_entropy(g, t, reduction='none')
            
            # Optional class balancing
            if self.balance_classes:
                # Weight minority class more heavily
                p = t.mean()
                weights = t / (2 * p + self.eps) + (1 - t) / (2 * (1 - p + self.eps))
                prop_loss = (prop_loss * weights).mean()
            else:
                prop_loss = prop_loss.mean()
        
        # 3. PEHE regularization (treatment effect variance)
        pehe_reg = ((y1_hat - y0_hat) ** 2).mean()
        
        # 4. Targeted regularization (TarReg)
        tar_reg = torch.tensor(0.0, device=y.device)
        if self.lambda_tar > 0 and self.epsilon is not None:
            # Compute IPW weights
            h = t / (g + self.eps) - (1 - t) / (1 - g + self.eps)
            
            # Factual predictions
            y_pred = t * y1_hat + (1 - t) * y0_hat
            
            # Perturbed predictions
            y_pert = y_pred + self.epsilon * h
            
            # Targeted regularization loss
            tar_reg = ((y - y_pert) ** 2).mean()
        
        # Total loss
        total_loss = (factual_loss + 
                     self.alpha * prop_loss + 
                     self.lambda_reg * pehe_reg + 
                     self.lambda_tar * tar_reg)
        
        if return_components:
            return {
                'total_loss': total_loss,
                'factual_loss': factual_loss,
                'propensity_loss': prop_loss,
                'pehe_reg': pehe_reg,
                'targeted_reg': tar_reg,
                'epsilon': self.epsilon.item() if self.epsilon is not None else 0.0
            }
        
        return total_loss

class AdaptiveDragonLoss(DragonLoss):
    """
    Adaptive version that adjusts loss weights during training based on 
    relative magnitudes of loss components.
    """
    def __init__(self, *args, adaptive_weights=True, momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_weights = adaptive_weights
        self.momentum = momentum
        
        # Running averages for adaptive weighting
        self.register_buffer('factual_avg', torch.tensor(1.0))
        self.register_buffer('prop_avg', torch.tensor(1.0))
        
    def forward(self, y0_hat, y1_hat, g_logits, t, y, return_components=False):
        # Get individual loss components
        components = super().forward(y0_hat, y1_hat, g_logits, t, y, return_components=True)
        
        if self.adaptive_weights and self.training:
            # Update running averages
            with torch.no_grad():
                self.factual_avg = (self.momentum * self.factual_avg + 
                                  (1 - self.momentum) * components['factual_loss'].detach())
                self.prop_avg = (self.momentum * self.prop_avg + 
                               (1 - self.momentum) * components['propensity_loss'].detach())
                
                # Adaptive alpha to balance factual and propensity losses
                ratio = self.factual_avg / (self.prop_avg + self.eps)
                adaptive_alpha = torch.clamp(ratio * self.alpha, 0.1, 2.0)
        else:
            adaptive_alpha = self.alpha
            
        # Recompute total loss with adaptive weights
        total_loss = (components['factual_loss'] + 
                     adaptive_alpha * components['propensity_loss'] + 
                     self.lambda_reg * components['pehe_reg'] + 
                     self.lambda_tar * components['targeted_reg'])
        
        if return_components:
            components['total_loss'] = total_loss
            components['adaptive_alpha'] = adaptive_alpha.item() if hasattr(adaptive_alpha, 'item') else adaptive_alpha
            return components
        
        return total_loss

