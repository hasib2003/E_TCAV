
import argparse

def args_linearity_check():

    parser = argparse.ArgumentParser(description='Evaluate the linearity of model')
    
    parser.add_argument('--model', type=str, default='resnet18', help='Model to be used, must be defined in utils.models.get_model func')
    parser.add_argument('--data_dir', type=str,help='path to directory containing dataset')
    parser.add_argument('--checkpoint', type=str,required=True, help='Checkpoint path to load model from')
    parser.add_argument('--save_dir', type=str,required=True, help='Dir to save results')
    
    parser.add_argument(
        '--start_module',
        type=str,
        help='top level module in target model to enforce linearity from'
    )


    return parser.parse_args()  
