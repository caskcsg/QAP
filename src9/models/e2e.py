import torch
from torch import nn
from src9.models.e2e_t import MME2E_T
import torch.nn.functional as F
from src9.models.transformer_encoder import WrappedTransformerEncoder
from torchvision import transforms
from facenet_pytorch import MTCNN
from src9.models.vgg_block import VggBasicBlock
from src9.complex import *
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class my_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim=None):
        super(my_Attention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        #print(self.all_head_size)
        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
        #         #print(250)
        #         nn.init.kaiming_normal_(m.weight)
    def transpose_for_scores(self, x):
        # print(type(x))
        # print(x.shape)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        # print(mixed_query_layer.shape)
        # print(mixed_key_layer.shape)
        # print(mixed_value_layer.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # print(query_layer.shape)
        # print(key_layer.shape)
        # print(value_layer.shape)
        # exit(0)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is 
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class my_TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead=1, dim_feedforward=128, dropout=0.1):
        super(my_TransformerLayer, self).__init__()
        self.self_attention = my_Attention(hidden_size, nhead, dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, hidden_size))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         #print(250)
        #         nn.init.kaiming_normal_(m.weight)
    def forward(self, src, attention_mask=None):
        src_1 = self.self_attention(src, src, attention_mask=attention_mask)
        src = src + self.dropout1(src_1)
        src = self.norm1(src)
        src_2 = self.fc(src)
        src = src + self.dropout2(src_2)
        src = self.norm2(src)

        return src

class my_TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(my_TransformerEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)
    def forward(self, src, attention_mask=None):
        for layer in self.layers:
            new_src = layer(src, attention_mask=attention_mask)
            src = src + new_src
        return src

# class my_Transformer(nn.Module):
#     def __init__(self, d_model, num_layers=1, nhead=1, dropout=0.1, dim_feedforward=128, max_seq_length=5000):
#         super(my_Transformer, self).__init__()
#         self.d_model = d_model
#         self.pos_encoder = nn.Embedding(max_seq_length, d_model)
#         self.encoder = my_TransformerEncoder(my_TransformerLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout), num_layers=num_layers)
#         self.decoder = nn.Linear(d_model, 1)
#         self.norm = nn.LayerNorm(d_model)
#         # for m in self.modules():
#         #     if isinstance(m, (nn.Conv2d, nn.Linear)):
#         #         #print(250)
#         #         nn.init.kaiming_normal_(m.weight)
#     def forward(self, input1, lens, attention_mask=None):
#         max_len = max(lens)
#         # print(max_len)
#         # print(lens[0])
#         mask = [([False] * (l + 1) + [True] * (max_len - l)) for l in lens]
#         mask = torch.tensor(mask).to(device=inputs.device)
#         inputs = list(inputs.split(lens, dim=0))
#         inputs = [padTensor(inp, max_len) for inp in inputs]
#         inputs = torch.stack(inputs, dim=0)


#         seq_length = input1.size()[1]
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=input1.device)
#         positions_embedding = self.pos_encoder(position_ids).unsqueeze(0).expand(input1.size()) # (seq_length, d_model) => (batch_size, seq_length, d_model)
#         input1 = input1 + positions_embedding
#         input1 = self.norm(input1)
#         hidden = self.encoder(input1, attention_mask=attention_mask)
#         out = self.decoder(hidden) # (batch_size, seq_len, hidden_dim)
#         out = (out[:,0,:], out, hidden) # ([CLS] token embedding, full output, last hidden layer)
#         return out
class Attention_1(nn.Module):#向量改为矩阵的
    def __init__(self, KV_size, num_attention_heads, attention_probs, Q_size, batchsize=16, trans_dim=100):
        super().__init__()
        if KV_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (KV_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(KV_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        #print(self.all_head_size)
        # visual_dim = 2048
        # if ctx_dim is None:
        #     ctx_dim = hidden_size
        # self.query = nn.Linear(hidden_size, self.all_head_size)
        # self.key = nn.Linear(ctx_dim, self.all_head_size)
        # self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.query = linear_c(Q_size, batchsize, trans_dim)
        self.key = linear_c(KV_size, batchsize, trans_dim)
        self.value = linear_c(KV_size, batchsize, trans_dim)

        self.dropout = nn.Dropout(attention_probs)
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
        #         #print(250)
        #         nn.init.kaiming_normal_(m.weight)
    def transpose_for_scores(self, x):
        # print(type(x))
        # print(x.shape)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def cal_score(self, x, y):
        # print(x.shape)
        # print(y.shape)
        # exit(0)

        x_f=torch.flatten(x, start_dim = -2, end_dim = -1)
        y_f=torch.flatten(y, start_dim = -2, end_dim = -1)
        #print(x_f.shape)
        #print(y_f.shape)
        result=torch.matmul(x_f,y_f.transpose(-1,-2))
        #print(result.shape)
        #exit(0)
        result=result.real
        x_imag=x.imag
        y_imag=y.imag
        result_bu=torch.matmul(torch.flatten(x_imag, start_dim = -2, end_dim = -1),torch.flatten(y_imag, start_dim = -2, end_dim = -1).transpose(-1,-2))
        output_real=result+2*result_bu

        return output_real
    def forward(self, KV, Q, K_mask, Q_mask):
        #print(hidden_states.shape)
        mixed_query_layer = self.query(Q)
        mixed_key_layer = self.key(KV)
        mixed_value_layer = self.value(KV)
        #print(mixed_query_layer.shape)
        # mixed_query_layer = hidden_states
        # mixed_key_layer = context
        # mixed_value_layer = context
        batch_size = mixed_key_layer.shape[0]
        seq_len = mixed_key_layer.shape[1]
        embedding_size = mixed_key_layer.shape[2]
        # query_layer = self.transpose_for_scores(mixed_query_layer)
        # key_layer = self.transpose_for_scores(mixed_key_layer)
        # value_layer = self.transpose_for_scores(mixed_value_layer)

        
        # print(mixed_query_layer[0].shape)
        # print(mixed_key_layer[0].shape)
        # print(mixed_value_layer[0].shape)
        # print(mixed_query_layer[1].shape)
        # print(mixed_key_layer[1].shape)
        # print(mixed_value_layer[1].shape)

        # for i in range(len(mixed_key_layer[1])):
        #     for j in range(50):
        #         if float(mixed_key_layer[1][i][j].trace())!=0:
        #             print(float(mixed_key_layer[1][i][j]))
        # exit(0)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print(250)
        # print(mixed_query_layer.shape)
        # print(mixed_key_layer.shape)
        attention_scores = self.cal_score(mixed_query_layer, mixed_key_layer)
        # print(attention_scores.shape)
        # exit(0)
        #attention_scores = torch.randn(mixed_key_layer.shape[0],50,50).to(device)
        # print(attention_scores.shape)
        # exit(0)
        # Apply the attention mask is 
        if K_mask is not None:
            #print(K_mask)
            K_mask = (1-K_mask)*(-100000)
            K_mask = K_mask.unsqueeze(1)
            attention_scores = attention_scores + K_mask
            #exit(0)
            

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        value_layer1 = mixed_value_layer.reshape(batch_size, seq_len, embedding_size*embedding_size)
        V_real=value_layer1.real
        V_imag=value_layer1.imag
        #value_layer1_imag = mixed_value_layer.reshape(batch_size, seq_len, embedding_size*embedding_size)
        # print(value_layer1.shape)
        # print(attention_probs.shape)
        # print(value_layer1[1][1])
        # print(attention_probs[1][1])
        context_layer_real = torch.matmul(attention_probs, V_real)
        context_layer_imag = torch.matmul(attention_probs, V_imag)
        if Q_mask is not None:
            #Q_mask = (1-K_mask)*(-100000)
            Q_mask = Q_mask.unsqueeze(-1)
            # print(Q_mask)
            # exit(0)
            context_layer_real = context_layer_real * Q_mask
            context_layer_imag = context_layer_imag * Q_mask
        # print(context_layer_real.shape)
        # print(context_layer_imag.shape)
        # exit(0)
        context_layer1 = torch.complex(context_layer_real,context_layer_imag)


        context_layer1 = context_layer1.reshape(batch_size, seq_len, embedding_size, embedding_size)
        #context_layer_imag = context_layer_imag.reshape(batch_size, seq_len, embedding_size, embedding_size)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        output=context_layer1
        return output

class my_attention1_1(nn.Module):#用text做key和value，用audio做q，做注意力，然后做自注意力，把结果再和文本做注意力，然后然后拼接高级语音特征进行分类
    def __init__(self, KV_size, Q_size, nhead=1, dropout=0.01, batchsize=16,trans_dim=100):
        super(my_attention1_1, self).__init__()
        self.context_cross_attention = Attention_1(KV_size, nhead, dropout, Q_size=Q_size, batchsize=batchsize, trans_dim=trans_dim)

        #self.fc1 = nn.Sequential(linear_c(context_size,batchsize,50), nn.ReLU(), linear_c(context_size,batchsize,50))
        #self.fc1 = nn.Sequential(linear_c(context_size,batchsize,50),QActivation(),linear_c(context_size,batchsize,50))
        self.fc1 =linear_c(Q_size,batchsize,trans_dim)
        # self.norm1 = QNorm(context_size)
        # self.norm2 = QNorm(context_size)
        self.dropout1 = QDropout(dropout)

    def forward(self, KV, Q, K_mask, Q_mask):
        # text = (src)
        # audio = (context)

        new_src = self.context_cross_attention(KV, Q, K_mask, Q_mask)
        #cross_src = new_src+audio
        # cross_src = [0.5*new_src[0]+0.5*audio[0],0.5*new_src[1]+0.5*audio[1]]
        # print(new_src[1][1])
        # exit(0)
        cross_src = 0.5*new_src+0.5*Q
        # print(cross_src[0].shape)
        # # print(cross_src[1].shape)
        # list1=[]
        # list2=[]
        # for i in range(len(cross_src[0])):
        #     for j in range(len(cross_src[0][i])):
        #         list1.append(float(cross_src[0][i][j].trace()))
        #         list2.append(float(cross_src[1][i][j].trace()))
        # print(list1)
        # print(list2)
        # # exit(0)
        # print(torch.diagonal(cross_src[0],0,2,3))
        # #print(torch.diagonal(cross_src[1],0,2,3))
        #cross_src = self.norm1(cross_src)
        # print(torch.diagonal(cross_src[0],0,2,3))
        # list1=[]
        # list2=[]
        # for i in range(len(cross_src[0])):
        #     for j in range(len(cross_src[0][i])):
        #         list1.append(float(cross_src[0][i][j].trace()))
        #         list2.append(float(cross_src[1][i][j].trace()))
        # print(list1)
        # print(list2)
        # #print(torch.diagonal(cross_src[1],0,2,3))
        # exit(0)
        #cross_src_1 = self.self_attention(cross_src, cross_src, attention_mask=attention_mask)
        cross_src_1 = self.fc1(cross_src)
        # cross_src2 = [0.5*cross_src[0] + 0.5*self.dropout1(cross_src_1)[0],0.5*cross_src[1] + 0.5*self.dropout1(cross_src_1)[1]]
        cross_src2 = 0.5*cross_src+0.5*self.dropout1(cross_src_1)
        #cross_src2 = self.norm2(cross_src2)

        return cross_src2

class MyCSA1_1(nn.Module):
    def __init__(self, KV_size, Q_size, nhead, dropout, batchsize, num_layers, trans_dim):
        super(MyCSA1_1, self).__init__()
        # self.layer1 = my_attention1_1(text_size, audio_size, nhead, dropout, batchsize)
        # self.layer2 = my_attention1_1(text_size, audio_size, nhead, dropout, batchsize)
        self.num_layers=num_layers
        self.layer1 = nn.ModuleList([my_attention1_1(KV_size, Q_size, nhead, dropout, batchsize, trans_dim) for i in range(self.num_layers)]) 
        # self.layer3 = my_attention1_1(text_size, audio_size, nhead, dropout)
        # self.layer4 = my_attention1_1(text_size, audio_size, nhead, dropout)
        # self.layer5 = my_attention1_1(text_size, audio_size, nhead, dropout)
        #self.layer6 = my_attention1(text_size, audio_size, nhead, dropout)
    def forward(self, KV, Q, K_mask=None, Q_mask=None):
        rr = Q
        # rr1 = self.layer1(src,rr,attention_mask=attention_mask)
        # rr2 = self.layer2(src,rr1,attention_mask=attention_mask)
        # rr2 = self.layer3(src,rr2,attention_mask=attention_mask)
        # rr2 = self.layer4(src,rr2,attention_mask=attention_mask)
        # rr2 = self.layer5(src,rr2,attention_mask=attention_mask)
        #rr2 = self.layer6(src,rr2,attention_mask=attention_mask)
        for i in range(self.num_layers):
            #print(250)
            rr = self.layer1[i](KV,rr,K_mask,Q_mask)
        return rr
class MME2E(nn.Module):
    def __init__(self, args,device):
        #print(device)
        super(MME2E, self).__init__()
        self.num_classes = args['num_emotions']
        self.args = args
        self.mod = args['modalities'].lower()
        self.device = device
        self.feature_dim = args['feature_dim']
        nlayers = args['trans_nlayers']
        nheads = args['trans_nheads']
        trans_dim = args['trans_dim']

        CMT_nheads = args['CMT_nheads']
        CMT_layers = args['CMT_layers']
        CMT_dropout = args['CMT_dropout']
        self.batchsize = args['batch_size']
        self.seq_len = args['text_max_len']
        text_cls_dim = 768
        if args['text_model_size'] == 'large':
            text_cls_dim = 1024
        if args['text_model_size'] == 'xlarge':
            text_cls_dim = 2048

        self.T = MME2E_T(feature_dim=self.feature_dim, size=args['text_model_size'])

        self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)
        self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

        self.V = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.A = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.v_flatten = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.a_flatten = nn.Sequential(
            nn.Linear(512 * 8 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.v_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)
        self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)

        self.v_out = nn.Linear(trans_dim, self.num_classes)
        self.t_out = nn.Linear(text_cls_dim, self.num_classes)
        self.a_out = nn.Linear(trans_dim, self.num_classes)

        self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)

        self.ImageEmbedding = ImageEmbedding(trans_dim)

        self.audio_encoder = WrappedTransformerEncoder(dim=1582, num_layers=nlayers, num_heads=nheads)
        self.video_encoder = WrappedTransformerEncoder(dim=711, num_layers=nlayers, num_heads=nheads)

        
        self.fc_t1 = nn.Linear(768,trans_dim)
        self.fc_a1 = nn.Linear(1582,trans_dim)
        self.fc_v1 = nn.Linear(711,trans_dim)

        self.att1 = MyCSA1_1(self.seq_len, self.seq_len, CMT_nheads, CMT_dropout, self.batchsize, num_layers=CMT_layers, trans_dim=trans_dim)
        self.att2 = MyCSA1_1(self.seq_len, self.seq_len, CMT_nheads, CMT_dropout, self.batchsize, num_layers=CMT_layers, trans_dim=trans_dim)

        self.att3 = MyCSA1_1(self.seq_len, self.seq_len, CMT_nheads, CMT_dropout, self.batchsize, num_layers=CMT_layers, trans_dim=trans_dim)
        self.att4 = MyCSA1_1(self.seq_len, self.seq_len, CMT_nheads, CMT_dropout, self.batchsize, num_layers=CMT_layers, trans_dim=trans_dim)        

        self.measurement1 = QMeasurement(trans_dim)
        self.measurement2 = QMeasurement(trans_dim)
        self.fc3 = nn.Sequential(nn.Linear(trans_dim, int(trans_dim/2)), 
                nn.ReLU(), 
                nn.Dropout(CMT_dropout), 
                nn.Linear(int(trans_dim/2), self.num_classes))
        #self.fc3 = nn.Linear(trans_dim, int(trans_dim/2))

        self.fc4 = nn.Linear(trans_dim,1)
        self.fc5 = nn.Linear(trans_dim,1)

    def forward(self, imgs, imgs_lens, specs, spec_lens, text, video_feature, audio_feature, senti):
        all_logits = []


        # print(video_feature.shape)
        # print(audio_feature.shape)
        # exit(0)
        if 't' in self.mod:
            # print(type(text))
            # print(len(text))
            # print(text)
            # print(**text)
            # exit(0)
            text_mask = text['attention_mask']
            text_cls = self.T(text, get_cls=True)

            text_imag = self.ImageEmbedding(senti)
            # print(text_mask.shape)
            # print(text_cls.shape)
            # print(text_imag.shape)
            # exit(0)
            text_cls = self.fc_t1(text_cls)
            #print(text_cls.device)
            #all_logits.append(self.t_out(text_cls))
        # print(all_logits[0].shape)
        # exit(0)
        if 'v' in self.mod:
            #print(imgs.shape)
            faces = self.mtcnn(imgs)
            # print(len(faces))
            # for i in range(len(faces)):
            #     if faces[i] is None:
            #         continue
            #     print(faces[i].shape)
            # exit(0)
            j=0
            for i, face in enumerate(faces):
                if face is None:
                    #print(i)
                    j+=1
                    center = self.crop_img_center(torch.tensor(imgs[i]).permute(2, 0, 1))
                    faces[i] = center
            # print(j)
            # exit(0)
            # print(len(faces))
            # for  j in range(len(faces)):
            #     print(faces[j].shape)
            #     if j==10:
            #         break
            faces = [self.normalize(face) for face in faces]
            # print(len(faces))
            # for  j in range(len(faces)):
            #     print(faces[j].shape)
            #     if j==10:
            #         break 
            # exit(0)           
            faces = torch.stack(faces, dim=0).to(self.device)
            #print(faces.device)
            faces = self.V(faces)
            #print(faces.shape)
            faces = self.v_flatten(faces.flatten(start_dim=1))
            #print(faces.shape)
            faces,input_mask, video_mask = self.v_transformer(faces, imgs_lens, get_cls=True,seq_len=self.seq_len)
            #print(faces.shape)
            #faces = self.fc_t1(faces)
            #print(video_feature.shape)
            video_imag,_,_=self.video_encoder(video_feature,imgs_lens,get_cls=True,seq_len=self.seq_len, input_mask=input_mask)


            # print(video_mask.shape)
            # print(video_mask)
            # exit(0)

            video_imag = self.fc_v1(video_imag)
            #all_logits.append(self.v_out(faces))

        if 'a' in self.mod:
            #print(specs.shape)
            for a_module in self.A:
                specs = a_module(specs)
                #print(specs.shape)

            specs = self.a_flatten(specs.flatten(start_dim=1))
            specs,input_mask,audio_mask = self.a_transformer(specs, spec_lens, get_cls=True,seq_len=self.seq_len)
            # print(specs.shape)
            # exit(0)
            audio_imag,_,_ = self.audio_encoder(audio_feature,spec_lens,get_cls=True,seq_len=self.seq_len,input_mask=input_mask)
            audio_imag = self.fc_a1(audio_imag)
            # print(audio_imag.shape)
            # exit(0)
            #all_logits.append(self.a_out(specs))
        # print(text_cls.shape)
        # print(text_imag.shape)

        #text_cls = F.normalize(text_cls, p=2, dim=-1, eps=1e-12)
        #specs = F.normalize(specs, p=2, dim=-1, eps=1e-12)
        #faces = F.normalize(faces, p=2, dim=-1, eps=1e-12)
        text_c = ComplexMultiply(text_cls, text_imag)#复向量
        audio_c = ComplexMultiply(specs, audio_imag)
        video_c = ComplexMultiply(faces, video_imag)
        # print(text_c[0].shape)
        # print(audio_c[0].shape)
        # print(video_c[0].shape)
        # print(text_c[1].shape)
        # print(audio_c[1].shape)
        # print(video_c[1].shape)
        # exit(0)
        text_cm = Qouter(text_c)#复矩阵
        audio_cm = Qouter(audio_c)
        video_cm = Qouter(video_c)
        # print(text_cm.shape)
        # print(audio_cm.shape)
        # print(video_cm.shape)
        # exit(0)
        t_a = self.att1(text_cm, audio_cm, text_mask, audio_mask)
        #print(text_audio.shape)
        t_a_v = self.att2(t_a, video_cm, audio_mask, video_mask)

        t_v = self.att3(text_cm, video_cm, text_mask, video_mask)
        t_v_a = self.att4(t_v, audio_cm, video_mask, audio_mask)

        rr1 = t_a_v[:,0]
        rr2 = t_v_a[:,0]
        len1=len(rr1)
        len2=len(rr1[0])
        score1 = self.measurement1(rr1)
        score2 = self.measurement1(rr2)
        score1 = self.fc4(score1)
        score2 = self.fc5(score2)
        scores = torch.cat((score1,score2),dim=-1)
        scores = nn.Softmax(dim=-1)(scores)
        alpha = scores[:,0]
        beta = scores[:,1]
        alpha = alpha.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        rr = (alpha*(torch.flatten(rr1,start_dim = -2, end_dim = -1))+beta*(torch.flatten(rr2,start_dim = -2, end_dim = -1))).reshape(len1,len2,len2)

        # rr = self.measurement2(t_a_v[:,0])
        rr = self.measurement2(rr)
        rr = self.fc3(rr)
        return rr
        #return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)

    def crop_img_center(self, img: torch.tensor, target_size=48):
        '''
        Some images have un-detectable faces,
        to make the training goes normally,
        for those images, we crop the center part,
        which highly likely contains the face or part of the face.

        @img - (channel, height, width)
        '''
        current_size = img.size(1)
        off = (current_size - target_size) // 2 # offset
        cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
        return cropped
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
