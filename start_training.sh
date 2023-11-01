source venv_test/bin/activate
python main.py --cycle=2 --is_train="no" --filepath=data/data.csv --learning_rate=0.01 --training_percentage=0.7 --validate_percentage=0.1 --batch_size=40 --input_size=6 --hidden_size=10 --patience=10 --num_of_epochs=100 --lossfunction="MSE"
deactivate