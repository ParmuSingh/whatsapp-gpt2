from transformers import pipeline, set_seed, GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, modelName='gpt2-medium', loadModel=False, alias = 'whatsapp-apurva'):

        # load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(modelName)
        print(f"eos_token: {self.tokenizer.eos_token}")
        print(f"bos_token: {self.tokenizer.bos_token}")
        print(f"unk_token: {self.tokenizer.unk_token}")
        print(f"pad_token: {self.tokenizer.pad_token}")
        print(f"mask_token: {self.tokenizer.mask_token}")

        # load model
        if loadModel:
            self.loadModel(alias)
        else:
            self.generator = TFGPT2LMHeadModel.from_pretrained(modelName)
        set_seed(42)

    def saveModel(self, alias):
        self.generator.save_pretrained(f'./fine-tuned-models/{alias}', save_optimizer_state=True)
        print('model saved')

    def loadModel(self, alias):
        self.generator = TFGPT2LMHeadModel.from_pretrained(f'./fine-tuned-models/{alias}')
        print('model loaded')

    def tokenizeFunctionDF(self, textDF):
        return self.tokenizer(textDF['dataset'], truncation=False, return_tensors='tf')

    def tokenizeFunction(self, text):
        return self.tokenizer(text, truncation=False, return_tensors='tf')

    def getModel(self):
        return self.generator

    def getTokenizer(self):
        return self.tokenizer

    def generate(self, seed_text):
        input_ids = self.tokenizer.encode(seed_text, return_tensors='tf')
        output = self.generator(input_ids)
        generated_sequence = tf.argmax(output.logits, axis=-1) # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions
        generated_text = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=False)
        return generated_text

    def trainTF(self, dataset, num_epoch):
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)# definining our loss function
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)# defining our metric which we want to observe
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')# compiling the model
        
        self.generator.compile(optimizer=optimizer, loss=[loss, *[None] * self.generator.config.n_layer], metrics=[metric])
        
        history = self.generator.fit(dataset, epochs=num_epoch)

        return history