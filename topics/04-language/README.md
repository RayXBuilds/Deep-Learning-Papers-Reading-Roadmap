# 语言、序列与大模型

[全部专题](../README.md) · [首页](../../README.md) · [年份索引](../../docs/roadmaps/timeline.md)

word2vec → Seq2Seq → Attention → Transformer；再比较 ELMo、GPT、BERT、XLNet、T5，接续规模化、对齐与新架构。

## 中文精读与阅读顺序

- [word2vec：预测方向怎样决定一个词收到的监督？](notes/word2vec.md)
- [Seq2Seq：读完整句之后，译文如何逐词生成？](notes/seq2seq.md)
- [Bahdanau Attention：每一步解码，应该重新读取哪里？](notes/bahdanau-attention.md)
- [Transformer：Q、K、V 与信息可见性](notes/transformer.md)
- [ELMo · 同一个词怎样获得随句子变化的表示？](notes/elmo.md)
- [GPT-1 · 只读左侧的模型怎样完成句对分类？](notes/gpt-1.md)
- [BERT：双向上下文、遮盖输入与监督位置怎样配合？](notes/bert.md)
- [Transformer-XL · 分段计算怎样保留上段的语言线索？](notes/transformer-xl.md)
- [XLNet · 打乱预测顺序，怎样保留词序又防止偷看答案？](notes/xlnet.md)
- [T5 · 一段缺失文本如何变成可生成的标签？](notes/t5.md)
- [GPT-3：示例怎样在参数固定时改变回答？](notes/gpt-3.md)
- [Chinchilla：同一笔训练预算该买参数还是 token？](notes/chinchilla.md)
- [InstructGPT：示范、排序和策略更新各监督什么？](notes/instructgpt.md)
- [DPO：如何从两个回答的策略概率恢复偏好？](notes/dpo.md)
- [Mamba-2：如何用两种运算读取同一段历史？](notes/mamba-2.md)
- [DeepSeek-V3：路由器怎样兼顾专家分工和设备负载？](notes/deepseek-v3.md)
- [DeepSeek-R1：只评价最终答案，哪些采样轨迹会受到鼓励？](notes/deepseek-r1.md)
- [Qwen3：同一组权重怎样学会长思考与直接回答？](notes/qwen3.md)

## 原始资料

以下按历史文件名列出全部资料，文件字节保持不变。文件名可能缩写，不能据此推定版本或精读状态；同主题新笔记在上方，逐篇书目校订仍需核对原文。

- [Ask Me Anything Dynamic Memory Networks for Natural Language Processing](<papers/nlp/Ask%20Me%20Anything%20Dynamic%20Memory%20Networks%20for%20Natural%20Language%20Processing.pdf>)
- [Bag of Tricks for Efficient Text Classification](<papers/nlp/Bag%20of%20Tricks%20for%20Efficient%20Text%20Classification.pdf>)
- [Character-Aware Neural Language Models](<papers/nlp/Character-Aware%20Neural%20Language%20Models.pdf>)
- [Distributed Representations ofWords and Phrases and their Compositionality](<papers/nlp/Distributed%20Representations%20ofWords%20and%20Phrases%20and%20their%20Compositionality.pdf>)
- [Joint Learning of Words and Meaning Representations](<papers/nlp/Joint%20Learning%20of%20Words%20and%20Meaning%20Representations.pdf>)
- [Sequence to Sequence Learning with Neural Networks](<papers/nlp/Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.pdf>) · [同题精读](notes/seq2seq.md)
- [TOWARDS AI-COMPLETE QUESTION ANSWERING](<papers/nlp/TOWARDS%20AI-COMPLETE%20QUESTION%20ANSWERING.pdf>)
- [Teaching Machines to Read and Comprehend](<papers/nlp/Teaching%20Machines%20to%20Read%20and%20Comprehend.pdf>)
- [Very Deep Convolutional Networks](<papers/nlp/Very%20Deep%20Convolutional%20Networks.pdf>)
- [A Neural Conversational Model](<papers/recurrent-models/A%20Neural%20Conversational%20Model.pdf>)
- [Generating Sequences With](<papers/recurrent-models/Generating%20Sequences%20With.pdf>)
- [Learning Phrase Representations using RNN Encoder–Decoder](<papers/recurrent-models/Learning%20Phrase%20Representations%20using%20RNN%20Encoder%E2%80%93Decoder.pdf>)
- [NEURAL MACHINE TRANSLATION](<papers/recurrent-models/NEURAL%20MACHINE%20TRANSLATION.pdf>)
- [Achieving Human Parity in Conversational Speech Recognition](<papers/speech/Achieving%20Human%20Parity%20in%20Conversational%20Speech%20Recognition.pdf>)
- [Deep Speech 2_ End-to-End Speech Recognition in](<papers/speech/Deep%20Speech%202_%20End-to-End%20Speech%20Recognition%20in.pdf>)
- [Deep neural networks for acoustic modeling in speech recognition_ The shared views of four research groups](<papers/speech/Deep%20neural%20networks%20for%20acoustic%20modeling%20in%20speech%20recognition_%20The%20shared%20views%20of%20four%20research%20groups.pdf>)
- [Fast and Accurate Recurrent Neural Network Acoustic Models for Speech](<papers/speech/Fast%20and%20Accurate%20Recurrent%20Neural%20Network%20Acoustic%20Models%20for%20Speech.pdf>)
- [Speech recognition with deep recurrent neural networks](<papers/speech/Speech%20recognition%20with%20deep%20recurrent%20neural%20networks.pdf>)
- [Towards End-To-End Speech Recognition with Recurrent Neural Networks](<papers/speech/Towards%20End-To-End%20Speech%20Recognition%20with%20Recurrent%20Neural%20Networks.pdf>)
- [A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation](<papers/translation/A%20Character-Level%20Decoder%20without%20Explicit%20Segmentation%20for%20Neural%20Machine%20Translation.pdf>)
- [Addressing the Rare Word Problem in Neural Machine Translation](<papers/translation/Addressing%20the%20Rare%20Word%20Problem%20in%20Neural%20Machine%20Translation.pdf>)
- [Effective Approaches to Attention-based Neural Machine Translation](<papers/translation/Effective%20Approaches%20to%20Attention-based%20Neural%20Machine%20Translation.pdf>)
- [Fully Character-Level Neural Machine Translation without Explicit Segmentation](<papers/translation/Fully%20Character-Level%20Neural%20Machine%20Translation%20without%20Explicit%20Segmentation.pdf>)
- [Google’s Neural Machine Translation System Bridging the Gap](<papers/translation/Google%E2%80%99s%20Neural%20Machine%20Translation%20System%20Bridging%20the%20Gap.pdf>)
- [Neural Machine Translation of Rare Words with Subword Units](<papers/translation/Neural%20Machine%20Translation%20of%20Rare%20Words%20with%20Subword%20Units.pdf>)

## 状态与下一步

语音识别、稀有词与字符级翻译等历史链条仍待精读。

[实验指南与实测边界](../../experiments/README.md) · [迁移对照](../../docs/archive/path-migration.md)
