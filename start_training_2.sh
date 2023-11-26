source venv/bin/activate
is_train=true 
num_exps=2 
filepath=data/data.csv 
learning_rate=0.001
training_percentage=0.7
validate_percentage=0.1
batch_size=40
input_size=6
hidden_size=10
patience=10
num_of_epochs=100
lossfunction="MSE" 
model_folder_dir="./models" 
plot_folder_dir="./pic" 
test_model_directory="model_params20231112_002801_2.pkl"
if [ "$is_train" = true ]; then
   echo 'training, validation and test'
   python main.py --num_exps=$num_exps \
    --is_train \
    --filepath=$filepath \
    --learning_rate=$learning_rate \
    --training_percentage=$training_percentage \
    --validate_percentage=$validate_percentage \
    --batch_size=$batch_size \
    --input_size=$input_size \
    --hidden_size=$hidden_size \
    --patience=$patience \
    --num_of_epochs=$num_of_epochs \
    --lossfunction=$lossfunction \
    --model_folder_dir=$model_folder_dir \
    --plot_folder_dir=$plot_folder_dir
else
    echo 'only conduct testing'
    python main.py --num_exps=$num_exps \
    --filepath=$filepath \
    --learning_rate=$learning_rate \
    --training_percentage=$training_percentage \
    --validate_percentage=$validate_percentage \
    --batch_size=$batch_size \
    --input_size=$input_size \
    --hidden_size=$hidden_size \
    --patience=$patience \
    --num_of_epochs=$num_of_epochs \
    --lossfunction=$lossfunction \
    --model_folder_dir=$model_folder_dir \
    --plot_folder_dir=$plot_folder_dir \
    --test_model_directory=$test_model_directory
fi
deactivate