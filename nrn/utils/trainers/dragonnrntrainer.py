import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nrn.sklogic.causal.dragon_loss import DragonLoss

class DragonNRNTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        alpha=None,
        lambda_reg=None,
        use_brier=False,
        lambda_tar=None,
        epochs=200,
        early_stopping_patience=20,
        device=None,
        loss_fn=None,
        balance_classes=False,
        clip_propensity=True,
        epsilon_init=0.1
    ):
        self.model = model
        self.eps = epsilon_init
        self.alpha = alpha
        self.balance_classes = balance_classes
        self.lambda_reg = lambda_reg
        self.lambda_tar = lambda_tar
        self.use_brier = use_brier
        self.clip_propensity = clip_propensity
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FIXED: Properly configure the loss function with instance parameters
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            # Use the instance parameters, not hardcoded values
            self.loss_fn = DragonLoss(
                alpha=self.alpha or 0.5,
                lambda_reg=self.lambda_reg or 0.5,
                lambda_tar=self.lambda_tar or 0.0,
                use_brier=self.use_brier,
                clip_propensity=self.clip_propensity,
                balance_classes=self.balance_classes,
                epsilon_init=epsilon_init
            )
        
        self.model.to(self.device)
        
        # ADDED: Move loss function to device if it has parameters
        if hasattr(self.loss_fn, 'parameters'):
            self.loss_fn.to(self.device)

    def train(self, train_dl: DataLoader, val_dl: DataLoader = None):
        best_loss = float('inf')
        patience = 0
        best_state_dict = None

        for epoch in range(self.epochs):
            self.model.train()
            
            # ADDED: Ensure loss function is in training mode
            if hasattr(self.loss_fn, 'train'):
                self.loss_fn.train()
            
            epoch_loss = 0

            for batch in train_dl:
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)  
                y = batch['outcome'].to(self.device)    

                # Forward pass
                y0_pred, y1_pred, g_preds = self.model(x)
                
                
                # Compute loss
                loss = self.loss_fn(y0_pred, y1_pred, g_preds, t, y)
                
                # ADDED: Debug print to verify loss is changing
                # print(f"Batch loss: {loss.item():.6f}")

                self.optimizer.zero_grad()
                loss.backward()
                
                # ADDED: Optional gradient clipping for stability
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_dl)
            
            # Step scheduler if provided
            if self.scheduler:
                self.scheduler.step()

            # Validation and early stopping
            if val_dl:
                val_loss = self.evaluate(val_dl)
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state_dict = self.model.state_dict().copy()
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1} with best val loss: {best_loss:.6f}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_loss:.6f}")

        # Load best model if validation was used
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)
            print(f"Loaded best model with validation loss: {best_loss:.6f}")

    def evaluate(self, dl: DataLoader):
        self.model.eval()
        
        # ADDED: Ensure loss function is in evaluation mode
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()
            
        loss_total = 0
        
        with torch.no_grad():
            for batch in dl:
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)   

                y0_pred, y1_pred, g = self.model(x)


                loss = self.loss_fn(y0_pred, y1_pred, g, t, y)
                loss_total += loss.item()

        return loss_total / len(dl)
    
    # ADDED: Method to get loss components for debugging
    def get_loss_components(self, dl: DataLoader, num_batches=1):
        """Get detailed loss components for debugging."""
        self.model.eval()
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()
            
        components_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(dl):
                if i >= num_batches:
                    break
                    
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)

                y0_pred, y1_pred, g = self.model(x)
                
                # Get loss components
                components = self.loss_fn(y0_pred, y1_pred, g, t, y, return_components=True)
                components_list.append({k: v.item() if hasattr(v, 'item') else v 
                                      for k, v in components.items()})
        
        # Average components
        if components_list:
            avg_components = {}
            for key in components_list[0].keys():
                avg_components[key] = sum(comp[key] for comp in components_list) / len(components_list)
            return avg_components
        return None