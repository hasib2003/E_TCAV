    
import argparse

def args_train_with_concepts():

    parser = argparse.ArgumentParser(description='Train a resnet-18 with LCAV')
    
    
    parser.add_argument('--train_dir', type=str, default='data', help='Path to train data directory')
    parser.add_argument('--test_dir', type=str, default='data', help='Path to test data directory')
    parser.add_argument('--dataset_name', type=str,required=False, default=None, help='Path to test data directory')

    parser.add_argument('--pretrained', action="store_true", help="Loads imagenet weights")
    parser.add_argument('--freeze_backbone', action="store_true", help="Freeze every thing except fc")
    
    
    parser.add_argument('--model', type=str, default='resnet18', help='Model to be used, must be defined in utils.models.get_model func')
   
    parser.add_argument('--concepts_dir', type=str, help='Path to dir where the concepts are defined')
    parser.add_argument('--concept_config_train', type=str, help='Path to file containing concepts configuration required during training')
    parser.add_argument('--concept_config_test', required=False, type=str, help='Path to file containing concepts configuration used to generated evaluation reports (Default concept_config_train)')
    
    parser.add_argument(
        '--train_activation_layers',
        nargs='+',
        default=["avgpool"],
        help='Activation layers to extract concepts from'
    )

    parser.add_argument(
        '--test_activation_layers',
        nargs='*',
        default=[],
        help='Activation layers to extract concepts from while testing'
    )


    parser.add_argument('--classifier', required=False, help='Classifier used in calculating the CAV')

    parser.add_argument('--checkpoint_dir', type=str, default='checkoints', help='Paht to dir to save checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, help='Checkpoint path to load model from')
 
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    parser.add_argument('--correction_frequency', type=int, default=2, help='Number of vanilla epochs after which correction epoch will be triggered through LCAV')


    return parser.parse_args()  

def args_train_base():

    parser = argparse.ArgumentParser(description='Train a resnet-18 with LCAV')
    
    
    parser.add_argument('--train_dir', type=str, required=False, help='Path to train data directory')
    parser.add_argument('--test_dir', type=str,required=False, help='Path to test data directory')
    parser.add_argument('--dataset_name', type=str,required=False, default='data', help='Path to test data directory')

    parser.add_argument('--checkpoint_dir', type=str, default='checkoints', help='Paht to dir to save checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, help='Checkpoint path to load model from')
    
    
    parser.add_argument('--model', type=str, default='resnet18', help='Model to be used, must be defined in utils.models.get_model func')
    parser.add_argument('--pretrained', action="store_true", help="Loads imagenet weights")
    parser.add_argument('--freeze_backbone', action="store_true", help="Freeze every thing except fc")
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')


    return parser.parse_args()  
