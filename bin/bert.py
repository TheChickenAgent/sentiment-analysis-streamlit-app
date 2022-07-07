import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm

class Bertenizer:
    """
    Bertenizer is a class that will BERTify the text passed to it.
    """
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """Initializer

        Args:
            model_name (str): in case a different BERT model from Huggingface would be preferred, pass it here.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertModel.from_pretrained(model_name)
    
    def bertify(self, text):
        """converts text to BERT embedding

        Args:
            text (str): text to be embedded

        Returns:
            transformers.modeling_tf_outputs.
            TFBaseModelOutputWithPoolingAndCrossAttentions: see https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel
        """
        encoded_input = self.tokenizer(text, return_tensors = 'tf', padding = True, truncation = True)
        output = self.model(encoded_input)
        return output

    def pooling(self, all_sentences: list, pooling_method: str):
        """Method that pools two sentences depending on the pooling method

        Args:
            allSentences (list): list of sentences that are going to be embedded
            pooling_method (str): if 'average', then average pooling, else max pooling

        Returns:
            v_total (tensor): pooled tensor with dimension (len(all_sentences), 768)
        """
        device_name = tf.test.gpu_device_name()
        if "GPU" not in device_name:
            print("GPU device not found")
        print('Found GPU at: {}'.format(device_name))

        with tf.device('/gpu:0'):
            v = self.bertify(all_sentences[0]).last_hidden_state #let the sentence through the BERT model and take the last_hidden_state
            if(pooling_method == 'average'):
                v_pooled = tf.nn.avg_pool(v, ksize = (v.shape[1],), strides = 1, padding = 'VALID') #average pool the (1, dim, 768) vector
            else:
                v_pooled = tf.nn.max_pool(v, ksize = (v.shape[1],), strides = 1, padding = 'VALID') #average pool the (1, dim, 768) vector
            v_total = tf.reshape(v_pooled, shape=(1, v.shape[0], v.shape[2])) #reshape it from (1,1,768) to (1, 768)

            for i in tqdm(range(1, len(all_sentences))):
                v = self.bertify(all_sentences[i]).last_hidden_state #let the sentence through the BERT model and take the last_hidden_state
                if(pooling_method == 'average'):
                    v_pooled = tf.nn.avg_pool(v, ksize = (v.shape[1],), strides = 1, padding = 'VALID') #average pool the (1, dim, 768) vector
                else:
                    v_pooled = tf.nn.max_pool(v, ksize = (v.shape[1],), strides = 1, padding = 'VALID') #average pool the (1, dim, 768) vector
                v_pooled = tf.reshape(v_pooled, shape=(1, v.shape[0], v.shape[2])) #reshape it from (1,1,768) to (1, 768)

                v_total = tf.concat([v_total, v_pooled], axis = 0) #concatenate it to the other vector(s)
        return v_total