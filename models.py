import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaEncoder, RobertaPooler, RobertaPreTrainedModel

class RobertaConfig(PretrainedConfig):
    model_type = "roberta"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

class RobertaModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        x
        # input_ids: torch.Tensor = None,
        # attention_mask: torch.Tensor = None,
        # token_type_ids: torch.Tensor = None,
        # position_ids: torch.Tensor = None,
        # output_attentions: bool = None,
        # output_hidden_states: bool = None
    ):
        input_ids, attention_mask, token_type_ids = x
        position_ids, output_attentions, output_hidden_states = None, None, None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
    
        head_mask               = self.get_head_mask(head_mask=None, num_hidden_layers=self.config.num_hidden_layers)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size())

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values_length=0,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return {
                'last_hidden_state': sequence_output,
                'pooler_output': pooled_output,
                # 'hidden_states': encoder_outputs.hidden_states,
                # 'attentions': encoder_outputs.attentions,
               }

class RobertaModelForSimCSE(RobertaModel):
    def __init__(self, *args, temperature=0.05, **kwargs):
        super().__init__(*args, **kwargs)

        # self.args = config
        self.temperature = temperature
        self.cosine_similarity = Similarity(temp=temperature)

    def create_label(self, x):

        if isinstance(x, dict):  # supervised
            batch_size = x["results1"][0].shape[0]
        else:   # unsupervised
            batch_size = x[0].shape[0]

        label = torch.arange(batch_size).long().to(self.device)

        return label
    
    def forward(self, x, phase:str = 'train'):

        if phase == 'train':

            if isinstance(x, dict):  # supervised

                r1 = super().forward(x['results1'])['pooler_output']
                r2 = super().forward(x['results2'])['pooler_output']
                r3 = super().forward(x['results3'])['pooler_output']
            
            else:  # unsupervised
                
                r1 = super().forward(x)['pooler_output']
                r2 = super().forward(x)['pooler_output']
                r3 = None
            
            cos_sim = self.cosine_similarity(r1.unsqueeze(1), r2.unsqueeze(0))
            if r3 is not None:
                r1_r3_cos = self.cosine_similarity(r1.unsqueeze(1), r3.unsqueeze(0))
                cos_sim = torch.cat([cos_sim, r1_r3_cos], axis=1)

        else:

            r1 = super().forward(x['results1'])['pooler_output']
            r2 = super().forward(x['results2'])['pooler_output']
            cos_sim = self.cosine_similarity(r1, r2)

        return cos_sim

class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()

        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) #/ self.temp
 