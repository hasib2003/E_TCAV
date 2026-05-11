
import argparse

def args_evaluate_tcav():

    parser = argparse.ArgumentParser(description='Evaluate the model based on concepts')

    parser.add_argument('--model', type=str, required=True, help='Model to be used, must be defined in utils.models.get_model func')
    
    parser.add_argument(
        '--layers',
        nargs='+',
        required=True,
        help='Activation layers to extract concepts from'
    )
    
    parser.add_argument('--tokenizer',required=False, type=str, help='Tokenizer to be used,if textual input, must be defined in utils.models.get_tokenizer func')
    parser.add_argument('--max_len',required=False, type=int, help='maximum sequence length required if it is the textual input')
    
    parser.add_argument("--input_type", choices=["text", "image"],help="Input type, either images or text",required=True)
    parser.add_argument('--concept_config', required=False, type=str, help='Path to file containing concepts configuration to be used for concept evaluations')
    
    parser.add_argument("--tcav_mode", choices=["default", "fast","both"],help="TCAV calculation method :: exact | approx | all",required=True)
    parser.add_argument('--classifier', type=str,required=True, help='Classifier used in calculating the CAV')

    parser.add_argument('--checkpoint', type=str,required=False, help='Checkpoint path to load model from')
    parser.add_argument('--save_dir', type=str,required=True, help='Dir to save results')
    parser.add_argument('--num_classes', type=int,required=False, help='No of classes, if None 1000 from Imagenet')
    

    return parser.parse_args()  
