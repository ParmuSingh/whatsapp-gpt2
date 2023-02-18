from chatminer.chatparsers import WhatsAppParser
import pandas as pd
import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(self, fileName):
        parser = WhatsAppParser(fileName)
        parser.parse_file()
        self.conversationDF = parser.parsed_messages.get_df()
        self.conversationDF['dataset'] = self.conversationDF['author'] + ": " + self.conversationDF['message']
        self.conversationDF.drop(['timestamp', 'author', 'message', 'weekday', 'hour', 'words', 'letters'], axis=1, inplace=True)

        # reverse rows cuz for some fuckall reason they're in reverse order
        self.conversationDF = self.conversationDF.reindex(index=self.conversationDF.index[::-1])
        self.conversationDF.reset_index(inplace=True, drop=True)
        print(self.conversationDF)

    def getTextDataset(self):
        return self.conversationDF
    
    def getTokenizedDataset(self):
        return self.tokenizedDataset
    
    def getTFDataset(self, datasetDF, tokenizer, limit = None, prep_large = False, BATCH_SIZE = 1):

        # the following code basically preps training data. sequence of 5 texts corresponds to 1 text that comes after the 5th one
        eos_token = 50256
        seq_length = 5 # 5 texts
        inputs, labels, inputSequences, seq = [], [], [], []
        i, seq_id = 0, 0

        longestInputSequenceLength = 0
        longestLabelSequenceLength = 0
        index = 0
        while index < len(datasetDF):
            row = datasetDF.loc[index]
            text = row["dataset"] + "\n" + tokenizer.eos_token
            seq_id+=1
            if seq_id % (seq_length + 1) == 0:

                # tokenize each text together into one sequence
                tokenizedSequences = tokenizer.encode(''.join(inputSequences))
                tokenizedLabel = tokenizer.encode(text)

                if len(tokenizedSequences) > longestInputSequenceLength:
                    longestInputSequenceLength = len(tokenizedSequences)

                if len(tokenizedLabel) > longestLabelSequenceLength:
                    longestLabelSequenceLength = len(tokenizedLabel)

                labels.append(tokenizedLabel)
                inputs.append(tokenizedSequences)
                inputSequences = []
                
                if prep_large:
                    index -= seq_length # to include last 4 texts of input sequence as 1st 4 texts of next sequence
                
                if limit != None:
                    limit -= 1
                    if limit == 0:
                        break
            else:
                inputSequences.append(text)
            index += 1
    
        # padding
        pad_token = tokenizer.encode(tokenizer.eos_token)[0] # using eos_token because for some fuckall reason pretrained gpt2 tokenizer doesn't have a pad_token.
        print(f"pad_token = {pad_token}")
        i=0
        while i < len(inputs):

            longestSequenceLength = max(longestInputSequenceLength, longestLabelSequenceLength)
            nPadTokensToAddInInput = longestSequenceLength - len(inputs[i])
            for _ in range(nPadTokensToAddInInput):
                inputs[i].append(pad_token)

            nPadTokensToAddInLabel = longestSequenceLength - len(labels[i])
            for _ in range(nPadTokensToAddInLabel):
                labels[i].append(pad_token)

            i += 1
        print(f"tokenizer.vocab_size: {tokenizer.vocab_size}")
        print(np.asarray(inputs).shape) # (10, 96) // when limit = 10
        print(np.asarray(labels).shape) # (10, 15)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        return dataset

