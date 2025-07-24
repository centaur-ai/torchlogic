import torch
from torch import nn
import logging

from ..nn import LukasiewiczChannelOrBlock, LukasiewiczChannelAndBlock, Predicates


class DragonNRNModule(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        layer_sizes,
        feature_names,
        n_selected_features_input,
        n_selected_features_internal,
        n_selected_features_output,
        perform_prune_quantile,
        ucb_scale,
        normal_form='dnf',
        add_negations=False,
        weight_init=0.2
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.normal_form = normal_form
        self.output_size = output_size
        self.layer_sizes = layer_sizes


        assert normal_form in ['dnf', 'cnf'], "'normal_form' must be 'dnf' or 'cnf'."
        assert len(layer_sizes) > 0, "layer_sizes must not be empty"

        # Input Layer - Start with AND for DNF, OR for CNF
        if normal_form == 'dnf':
            input_layer = LukasiewiczChannelAndBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=n_selected_features_input,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='layer_0',
                add_negations=add_negations,
                weight_init=weight_init
            )
        else:  # CNF
            input_layer = LukasiewiczChannelOrBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=n_selected_features_input,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='layer_0',
                add_negations=add_negations,
                weight_init=weight_init
            )

        model_layers = [input_layer]

        # Internal Layers - Proper alternation
        for i in range(1, len(layer_sizes)):
            prev_layer = model_layers[-1]
            outputs_key = f'layer_{i}'

            # Fixed alternation logic:
            # For DNF: AND -> OR -> AND -> OR ...
            # For CNF: OR -> AND -> OR -> AND ...
            if normal_form == 'dnf':
                # DNF: start with AND (i=0), then alternate OR (i=1), AND (i=2), etc.
                use_and_block = (i % 2 == 0)
            else:  # CNF
                # CNF: start with OR (i=0), then alternate AND (i=1), OR (i=2), etc.
                use_and_block = (i % 2 == 1)

            if use_and_block:
                block_cls = LukasiewiczChannelAndBlock
            else:
                block_cls = LukasiewiczChannelOrBlock

            internal_layer = block_cls(
                channels=output_size,
                in_features=layer_sizes[i - 1],
                out_features=layer_sizes[i],
                n_selected_features=n_selected_features_internal,
                parent_weights_dimension='out_features',
                operands=prev_layer,
                outputs_key=outputs_key,
                weight_init=weight_init
            )
            model_layers.append(internal_layer)

        # Shared Trunk
        self.shared_trunk = nn.Sequential(*model_layers)
        final_layer = model_layers[-1]
        final_features = layer_sizes[-1]

  
        self.head_f0 = LukasiewiczChannelAndBlock(
            channels=output_size,
            in_features=final_features,
            out_features=1,
            n_selected_features=n_selected_features_output,
            parent_weights_dimension='out_features',
            operands=final_layer,
            outputs_key='head_f0',
            weight_init=weight_init
        )

        self.head_f1 = LukasiewiczChannelAndBlock(
            channels=output_size,
            in_features=final_features,
            out_features=1,
            n_selected_features=n_selected_features_output,
            parent_weights_dimension='out_features',
            operands=final_layer,
            outputs_key='head_f1',
            weight_init=weight_init
        )

        self.head_g = LukasiewiczChannelAndBlock(
            channels=output_size,
            in_features=final_features,
            out_features=1,
            n_selected_features=n_selected_features_output,
            parent_weights_dimension='out_features',
            operands=final_layer,
            outputs_key='head_g',
            weight_init=weight_init
            )
 

        # Need to figure out how to handle pruning and UCB scaling
        self.perform_prune_quantile = perform_prune_quantile
        self.ucb_scale = ucb_scale

    def forward(self, x: torch.Tensor):
        trunk_output = self.shared_trunk(x)

        # Each head output shape: [B, C, F]
        q_t0 = self.head_f0(trunk_output).mean(dim=(1, 2))
        q_t1 = self.head_f1(trunk_output).mean(dim=(1, 2))
        g = self.head_g(trunk_output).mean(dim=(1, 2))
        #g = g.clamp(1e-9, 1 - 1e-9)  # Ensure g is in (0, 1)
        
        # Logging
        #print(f"q_t0 shape: {q_t0.shape}, q_t1 shape: {q_t1.shape}, g shape: {g.shape}")

        return q_t0, q_t1, g

    def get_layer_info(self):
        """Debug method to inspect layer structure"""
        info = []
        info.append(f"Normal form: {self.normal_form}")
        info.append(f"Layer sizes: {self.layer_sizes}")
        
        for i, layer in enumerate(self.shared_trunk):
            layer_type = "AND" if isinstance(layer, LukasiewiczChannelAndBlock) else "OR"
            info.append(f"Layer {i}: {layer_type} - {layer.in_features} -> {layer.out_features}")
        
        return "\n".join(info)