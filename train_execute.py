import argparse
import torch
from construct_network import SnnTrain


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = v.lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean (true/false).")


parser = argparse.ArgumentParser()
parser.add_argument('--set_seed', type=int, required=True)
parser.add_argument('--delay_parametrization', type=str, required=True)
parser.add_argument('--delay_order', type=int, required=True)
parser.add_argument('--h', type=int, required=True)
parser.add_argument('--delay_parametrization_trainable', type=str2bool, required=True)
parser.add_argument('--neuron_type', type=str, required=True)
args = parser.parse_args()

delay_parametrization = args.delay_parametrization  # 'rand', 'ones', "decay_exp" "decay_lin"
delay_order = args.delay_order
delay_parametrization_trainable = args.delay_parametrization_trainable
set_seed = args.set_seed
neuron_type = args.neuron_type
h = args.h

# delay_parametrization = 'ones'
# delay_order = 0
# delay_parametrization_trainable = False
# set_seed = 0
# neuron_type = 'adLIF'


if __name__ == "__main__":

    # For saving
    model_name = (f'nd{delay_order}__'
                  f'Ad{delay_parametrization}__'
                  f'AdTrain{delay_parametrization_trainable}__'
                  f'Neuron{neuron_type}__'
                  f'h{h}__'
                  f'Seed{set_seed}')

    print(f'MODEL NAME: {model_name}')
    save_file_path = f'/mimer/NOBACKUP/groups/snn/time_varying_project/saved_models/Delays/after_HPO/{model_name}.pkl'
    nb_epochs = 50

    # Initialize network
    model = SnnTrain(nb_input_neurons=140,  # binned
                     nb_hidden_neurons=h,
                     nb_output_neurons=20,
                     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                     nb_epochs=nb_epochs,

                     neuron_type=neuron_type,
                     delay_order=delay_order,
                     delay_parametrization=delay_parametrization,
                     delay_parametrization_trainable=delay_parametrization_trainable,
                     set_seed=set_seed,

                     # After HPO
                     batch_size=64,
                     dropout=0.6,
                     lr=0.01,
                     w_decay=0.0,

    )

    # Train network
    model.train_network(nb_epochs=nb_epochs,
                        save_file_path=save_file_path)
