import os
import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)   
    parser.add_argument('--patience', type=int, default=5, help='Early Stopping')
    parser.add_argument('--model_path', type=str, default='results/')
    parser.add_argument('--exp_num', default='1', type=str)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--pretrained_model', type=str, default='microsoft/graphcodebert-base')
    parser.add_argument('--simcse_model', type=str, default='../SimCSE/result/my-unsup-simcse-codebert-base/pytorch_model.bin')
                    
    args = parser.parse_args()

    return args    
    
if __name__ == '__main__':
    args = getConfig()
    args = vars(args)
    print(args)