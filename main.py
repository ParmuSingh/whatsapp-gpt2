from model import Model
from whatsappdataset import Dataset

MODEL_NAME = "gpt2-medium" # use gpt2 if you have enough memory to train model
MODEL_ALIAS = "whatsapp-apurva"
DATASET_FILE = "WhatsApp Chat with Apurva.txt"
RESUME_MODEL = True
TRAIN_MODEL = False
GENERATE_TEXT = True
EPOCHS = 1

if __name__ == '__main__':
     
    ## initialize model
    gpt2Model = Model(MODEL_NAME, loadModel = RESUME_MODEL, alias = MODEL_ALIAS)

    ## prepare data
    dataset = Dataset(DATASET_FILE)    
    textDatasetDF = dataset.getTextDataset()

    if TRAIN_MODEL:
        ## fine tune model on whatsapp chat
        tf_dataset = dataset.getTFDataset(datasetDF=textDatasetDF, tokenizer=gpt2Model.getTokenizer(), limit = 300, prep_large = True)
        history = gpt2Model.trainTF(tf_dataset, num_epoch=EPOCHS)
        print(history)
        gpt2Model.saveModel(alias = MODEL_ALIAS)

    if GENERATE_TEXT:
        ## create input text for GPT-2
        inputText = ""
        limit_texts = 5
        for ind in textDatasetDF.index:
            inputText+=f"{textDatasetDF['dataset'][ind]}\n"
            limit_texts -= 1
            if limit_texts == 0:
                break

        # inputText = inputText[len(inputText) - 1024:] # max sequence length is 1024

        ## generate
        print(f"\n\n---------------- start model ----------------")

        print(f"input text: {inputText}\n\ngenerated:")
        generatedToken = gpt2Model.generate(inputText)

        while generatedToken != gpt2Model.getTokenizer().eos_token:
            inputText += generatedToken
            inputText = inputText[len(generatedToken):]

            generatedToken = gpt2Model.generate(inputText)