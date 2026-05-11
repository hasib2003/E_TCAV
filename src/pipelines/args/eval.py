
import argparse

def args_evaluate_tcav():

    parser = argparse.ArgumentParser(description='Evaluate the model based on concepts')
    
    

    parser.add_argument('--model', type=str, default='resnet18', help='Model to be used, must be defined in utils.models.get_model func')
    parser.add_argument('--random_prefix', type=str, required=True, help='Model to be used, must be defined in utils.models.get_model func')
    parser.add_argument('--checkpoint', type=str,required=False, help='Checkpoint path to load model from')
    parser.add_argument('--save_dir', type=str,required=True, help='Dir to save results')
    parser.add_argument('--num_classes', type=int,required=False, help='No of classes, if None 1000 from Imagenet')
    parser.add_argument('--num_exps', type=int,required=True, help='No of experiments')
    

    parser.add_argument("--mode", choices=["exact", "approx","all"],help="TCAV calculation method :: exact | approx | all",required=True)
    parser.add_argument('--classifier', type=str,required=True, help='Classifier used in calculating the CAV')

    parser.add_argument('--concepts_dir', type=str, help='Path to dir where the concepts are defined')
    parser.add_argument('--concept_config', required=False, type=str, help='Path to file containing concepts configuration to be used for concept evaluations')
    
    parser.add_argument(
        '--target_layers',
        nargs='+',
        default=["avgpool"],
        help='Activation layers to extract concepts from'
    )


    return parser.parse_args()  
