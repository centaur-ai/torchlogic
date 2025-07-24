import numpy as np
import pandas as pd
import torch
from nrn.models.base.dragon import DragonNet
from nrn.utils.trainers.dragonnrntrainer import DragonNRNTrainer
from nrn.sklogic.causal.dragon_loss import DragonLoss,AdaptiveDragonLoss
from nrn.sklogic.base.base_estimator import BaseSKLogicEstimator
from nrn.sklogic.datasets.causal_dataset import CausalDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from nrn.utils.mic import compute_mic_matrix
from torch import optim


class DragonNRNRegressor(BaseSKLogicEstimator):
    def __init__(
        self,
        target_name=None,
        feature_names=None,
        treatment_col = 'treatment',
        layer_sizes=[8, 8],
        n_selected_features_input=4,
        n_selected_features_internal=4,
        n_selected_features_output=4,
        perform_prune_quantile=0.5,
        ucb_scale=1.5,
        add_negations=False,
        normal_form='dnf',
        weight_init=0.1,
        binarization=True,
        tree_num=10,
        tree_depth=5,
        tree_feature_selection=0.5,
        thresh_round=3,
        loss_fn=AdaptiveDragonLoss(),
        learning_rate=0.01,
        weight_decay=0.001,
        t_0=3,
        t_mult=4,
        epochs=200,
        batch_size=32,
        holdout_pct=0.2,
        early_stopping_plateau_count=20,
        evaluation_metric=mean_squared_error,
        pin_memory=False,
        persistent_workers=False,
        num_workers=0
    ):
        super().__init__(
            binarization=binarization,
            tree_num=tree_num,
            tree_depth=tree_depth,
            tree_feature_selection=tree_feature_selection,
            thresh_round=thresh_round,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            t_0=t_0,
            t_mult=t_mult,
            epochs=epochs,
            batch_size=batch_size,
            holdout_pct=holdout_pct,
            early_stopping_plateau_count=early_stopping_plateau_count,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers
        )

        self.target_name = target_name
        self.feature_names = feature_names
        self.treatment_col = treatment_col
        self.layer_sizes = layer_sizes
        self.n_selected_features_input = n_selected_features_input
        self.n_selected_features_internal = n_selected_features_internal
        self.n_selected_features_output = n_selected_features_output
        self.perform_prune_quantile = perform_prune_quantile
        self.ucb_scale = ucb_scale
        self.add_negations = add_negations
        self.normal_form = normal_form
        self.weight_init = weight_init
        self.loss_fn = loss_fn
        self.evaluation_metric = evaluation_metric

        # Separate scalers for targets and features
        self.target_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()

    def _fit_transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.target_scaler.fit_transform(y), columns=[self.target_name], dtype=np.float32)

    def _transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.target_scaler.transform(y), columns=[self.target_name])

    def _inverse_transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.target_scaler.inverse_transform(y), columns=[self.target_name])

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        if not isinstance(y, pd.DataFrame):
            y = self._handle_non_dataframe_targets(y)
        y = y.copy()
        y_arr = y.values.astype(np.float32).reshape(-1)

        # Pull treatment before fitting transformation
        if self.treatment_col not in X.columns:
            raise ValueError(f"'{self.treatment_col}' must be present in the input data.")
        t_arr = X[self.treatment_col].values.astype(np.float32)
        X_cov = X.drop(columns=[self.treatment_col])

        X_cov = self._handle_empty_feature_names(X_cov)

        X_cov = self._fit_transform_encode_data(X_cov)
        if self.binarization:
            X_cov = self._fit_transform_binarize_features(X_cov, y_arr > y.mean())

        # Fit the feature scaler on the processed features
        self.feature_scaler.fit(X_cov)

        feature_names = X_cov.columns  # pass these into DragonNet
        X_arr = X_cov.values
        

        # Build CausalDataset with (X_cov, t, y)
        dataset = CausalDataset(X_arr, t_arr, y_arr)
        train_dl, train_holdout_dl = self._generate_stratified_training_data_loaders(dataset, t_arr,0.27)#_generate_training_data_loaders(dataset)


        mic_c_policy, _ = compute_mic_matrix(X_cov, y, alpha=.45, c=6)
        mic_c_policy = torch.tensor(mic_c_policy)



        if X_cov.shape[1] < self.n_selected_features_input:
            print(f"Warning: The number of features ({X_cov.shape[1]}) is less than 'n_selected_features_input' ({self.n_selected_features_input}). Using number of features instead.")
            self.n_selected_features_input = X_cov.shape[1]

        self.model = DragonNet(
            input_size=X_cov.shape[1],
            layer_sizes=self.layer_sizes,
            feature_names=list(feature_names),
            n_selected_features_input=self.n_selected_features_input,
            n_selected_features_internal=self.n_selected_features_internal,
            n_selected_features_output=self.n_selected_features_output,
            perform_prune_quantile=self.perform_prune_quantile,
            ucb_scale=self.ucb_scale,
            normal_form=self.normal_form,
            add_negations=self.add_negations,
            weight_init=self.weight_init
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.t_0, T_mult=self.t_mult)

        trainer = DragonNRNTrainer(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=self.loss_fn,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_plateau_count
        )

        # Train model
        trainer.train(train_dl, train_holdout_dl)

    def _get_instance_params(self):
        """
        Get parameters for this model type.

        Returns:
            dict: parameters and values
        """
        return {
            'layer_sizes': self.layer_sizes,
            'n_selected_features_input': self.n_selected_features_input,
            'n_selected_features_internal': self.n_selected_features_internal,
            'n_selected_features_output': self.n_selected_features_output,
            'perform_prune_quantile': self.perform_prune_quantile,
            'ucb_scale': self.ucb_scale,
            'prune_strategy': self.prune_strategy,
            'delta': self.delta,
            'bootstrap': self.bootstrap,
            'swa': self.swa,
            'add_negations': self.add_negations,
            'weight_init': self.weight_init,
            'loss_fn': self.loss_fn,
            'perform_prune_plateau_count': self.perform_prune_plateau_count,
            'increase_prune_plateau_count': self.increase_prune_plateau_count,
            'increase_prune_plateau_count_plateau_count': self.increase_prune_plateau_count_plateau_count,
            'evaluation_metric': self.evaluation_metric,
        }

    def explain_sample(
            self,
            X: pd.DataFrame,
            sample_index: int = 0,
            quantile: float = 1.0
    ) -> str:
        """
        Generate a sample explanation

        Args:
            X (pd.DataFrame): DataFrame of input features.
            sample_index (int): Index of sample to explain.
            quantile (float): Percent of model to explain

        Returns:
            str: explanation for selected sample
        """
        self._validate_fit()
        
        # create textual feature names if they are not given by user
        if self.feature_names is None:
            self.feature_names = [f"the {x} was" for x in X.columns]
            X.columns = self.feature_names
        else:
            X.columns = self.feature_names

        X = self._encode_prediction_data(X)

        self.min_max_features_dict = {
            col: {'min': self.numeric_features.iloc[:, i].min(), 'max': self.numeric_features.iloc[:, i].max()}
            for i, col in enumerate(self.numeric_features.columns)
        }

        dataset = CausalDataset(X.values, np.ones(shape=(X.shape[0], 1)))

        return self.model.explain_samples(
            dataset[sample_index]['features'].unsqueeze(0),
            quantile=quantile,
            target_names=[self.target_name],
            explain_type='both',
            sample_explanation_prefix="The sample has",
            print_type='logical',
            ignore_uninformative=True,
            rounding_precision=3,
            show_bounds=not self.binarization,
            simplify=True,
            exclusions=None,
            inverse_transform_target=self.target_scaler.inverse_transform
        )

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.feature_scaler, 'scale_'):
            raise RuntimeError("Feature scaler not fitted. Did you forget to call fit()?")

        return pd.DataFrame(
            self.feature_scaler.transform(X),
            columns=X.columns,
            index=X.index
        )