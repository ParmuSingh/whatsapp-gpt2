# whatsapp-gpt2

Train a GPT2 language model on your whatsapp conversations. This repo uses a pretrained gpt2 model and fine tunes it with your whatsapp chat. To use your whatsapp chat for training, export your chat by following this guide: https://faq.whatsapp.com/1180414079177245/?cms_platform=android

Save the exported `WhatsApp Chat with [name].txt` file in root of this project, and set `DATASET_FILE` in `main.py` to tell the model to use that as training data.

# install dependencies:

`pip3 install tensorflow numpy chat-miner transformers datasets`

if using `tensorman`:
`tensorman run pip3 install tensorflow numpy chat-miner transformers datasets`

# Configure

open `main.py` and modify settings:
- `MODEL_NAME`: name of the model to use. Can be `gpt2`, `gpt2-medium`, `gpt2-xl`, etc.
- `MODEL_ALIAS`: alias for the model. Used to nomenclature when saving the model to disk.
- `DATASET_FILE`: name of the file that contains whatsapp chat
- `RESUME_MODEL`: boolean, if set to `True`, model is loaded from disk. Useful to resuming training.
- `TRAIN_MODEL`: boolean, if set to `True`, triggers fine-tuning.
- `GENERATE_TEXT`: boolean, if set to `True`, model will generate conversation based on the model.
- `EPOCHS`: number of epochs for fine-tuning the model

# Usage

run with `tensorman run --gpu python3 ./main.py` to use GPU acceleration. `tensorman` is only available on Linux.

# Limitations

- Training this model requires at least 16GB memory, that too on a reduced dataset.
- Trains on single whatsapp chat at once, however the model can be trained multiple times on multiple conversations.
- Text generation is WIP

# Use cases

- This model can be used to build a chatbot that mimicks the way you text people on whatsapp.
- Simulate a conversation similar to your whatsapp conversations

# Model Structure

- This model uses `<|endoftext|>` as `pad_token`.
- Trained by taking 5 texts and predicting 6th text.