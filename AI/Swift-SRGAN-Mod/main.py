import os
import sys
from Data import GenerateTrainingDataset

def main():
    
    given_input = input("\n1 - Test Swift-SRGAN\n2 - Train the Neural Network\n3 - Prepare a HR Dataset for training\n4 - Exit\n")
    
    exit = False
    while not exit:
        exit = True
        match given_input:
            case '1': # Test Swift-SRGAN
                pass
            
            case '2': # Train the Neural Network
                pass
            
            case '3': # Prepare a HR Dataset for training "./Datasets/DIV2K/Train/HR/DIV2K_train_HR"
                dataset_directory = input("\nType the dataset path: ")
                if os.path.isdir(dataset_directory):
                    GenerateTrainingDataset(dataset_directory).processDataset()
                    print("\nNew training Dataset can be found in ./Datasets/Current")
                else:
                    print("\nGiven path is invalid\n")
                    exit = False
                
            case '4': # Exit
                sys.exit("\nProgram closed sucessfully\n")
                
            case _:
                print("Given input was invalid\n\n\n")
                given_input = input("\n1 - Test Swift-SRGAN\n2 - Train the Neural Network\n3 - Prepare a HR Dataset for training\n4 - Exit\n")
                exit = False

if __name__ == "__main__":
    main()