# 整理的部分编程人员可能会用到的学习资料

## 学术类大模型

类别 | 名称 | 描述 | 链接  
---|---|---|---  
综合 | ChatGLM | 开源双语对话语言模型 |  [官网](https://models.aminer.cn/) [Github](https://github.com/THUDM/ChatGLM-6B)  
综合 | ChatGPT 学术优化 | 专为科研工作设计的 ChatGPT 扩展 | [官网]() [Github](https://github.com/binary-husky/chatgpt_academic)  
综合 | HuggingGPT | HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace |  [Github](https://github.com/microsoft/JARVIS) [paper](https://arxiv.org/abs/2303.17580)  

## 微调大模型

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
大模型微调 | Lora | Low-Rank Adaptation of Large Language Models |  [Web]() [Github](https://github.com/microsoft/LoRA) [Paper](https://arxiv.org/abs/2106.09685) [Colab]()  
大模型微调 | Prefix-Tuning | Dynamic Prefix-Tuning for Generative Template-based Event Extraction |  [Web]() [Github](https://github.com/XiangLi1999/PrefixTuning) [Paper](https://arxiv.org/abs/2205.06166) [Colab]()  
大模型微调 | P-Tuning | Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks |  [Web]() [Github](https://github.com/THUDM/P-tuning) [Paper](https://aclanthology.org/2022.acl-short.8/) [Colab]()  
大模型微调 | P-Tuning v2 | Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks |  [Web]() [Github](https://github.com/XiangLi1999/PrefixTuning) [Paper](https://arxiv.org/abs/2110.07602) [Colab]()  
  
## 代码生成模型

类别 | 名称 | 描述 | 链接  
---|---|---|---  
代码生成 | Copilot | Your AI pair programmer |[web](https://github.com/features/copilot/) [Github](https://github.com/github/copilot-docs) [paper]()  
代码生成 | CodeGeeX | CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X |  [web](https://codegeex.cn/zh-CN) [Github](https://github.com/THUDM/CodeGeeX) [paper](https://arxiv.org/abs/2303.17568)  
代码生成 | CodeGen | CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis |  [web](https://cloud.codegen.cc/) [Github](https://github.com/salesforce/CodeGen) [paper](https://arxiv.org/abs/2203.13474)  
代码生成 | InCoder | InCoder: A Generative Model for Code Infilling and Synthesis |  [web](https://sites.google.com/view/incoder-code-models) [Github](https://github.com/dpfried/incoder) [paper](https://arxiv.org/abs/2204.05999)  
  
## 语音转文字

类别 | 名称 | 描述 | 链接  
---|---|---|---  
语音转文字 | Google 语音识别 API | 【收费】Google Speech-to-Text API支持实时语音转文字、长时间的录音文件识别，以及多种语音编码格式。它可以在各种情境下，如电话、视频、语音助手等方面识别多达120种语言和方言。 |[官网](https://cloud.google.com/speech-to-text) [示例]()  
语音转文字 | 腾讯云语音识别 API |【收费】腾讯云语音识别API支持60种语言和多种语音编码格式。它可以实时转换语音为文字，并适用于电话、视频、语音助手等场景。 |[官网](https://www.tencentcloud.com/zh/products/asr) [示例]()  
语音转文字 | 百度语音识别 API |【收费】百度语音识别API支持实时语音识别、长时间的录音文件识别，以及多种语音编码格式。它支持普通话、粤语、四川话等多种方言，适用于电话、视频、语音助手等场景。|  [官网](https://cloud.tencent.com/product/asr) [示例]()  
语音转文字 | iFLYTEK 语音识别 API | 【收费】iFLYTEK是科大讯飞推出的一款语音识别服务。它支持多种语言，包括中文。在中文语音识别领域，iFLYTEK 的表现尤为出色。 |[官网](https://www.xfyun.cn/services/voicedictation) [示例]()  
语音转文字 | CMU Sphinx 语音识别 | CMU Sphinx 是一个免费、开源的语音识别系统，由卡内基梅隆大学开发。它支持多种语言，包括中文。|  [官网](https://cmusphinx.github.io/) [GitHub](https://github.com/cmusphinx)  
语音转文字 | Mozilla DeepSpeech 语音识别 | Mozilla DeepSpeech是一个免费、开源的基于深度学习的语音识别系统。DeepSpeech支持多种语言，包括中文。 |[官网](https://www.mozilla.org/en-US/firefox/voice/) [GitHub](https://github.com/mozilla/DeepSpeech)  
语音转文字 | Vosk-API 语音识别 | Vosk 是一个离线的开源语音识别系统，基于KaldiASR。它支持许多语言，包括中文，并且可以在不同平台上运行，如Linux、Windows、macOS、Android和iOS。 |[官网](https://alphacephei.com/vosk/) [GitHub](https://github.com/alphacep/vosk-api)  
  
## 分词、词性标注、依存句法分析

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
分词 | Jieba（结巴）中文分词 |结巴分词是一个非常流行的中文分词工具，提供基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图（DAG），并采用动态规划查找最大概率路径，找出基于词频的最大切分组合。|  [WEB]() [Github](https://github.com/fxsjy/jieba)  
基础 | SnowNLP |SnowNLP是一个Python写的中文自然语言处理库，可以方便地处理中文文本内容，提供了中文分词、词性标注、情感分析等功能。 |  [WEB]() [Github](https://github.com/isnowfy/snownlp)  
基础 | THULAC | 清华大学自然语言处理与社会人文计算实验室研制的一套中文词法分析工具包，具有中文分词和词性标注功能。 |[WEB](http://thulac.thunlp.org/) [Github](https://github.com/thunlp/THULAC-Python)  
基础 | LTP |由哈尔滨工业大学社会计算与信息检索研究中心开发的一套中文自然语言处理系统，提供了中文分词、词性标注、命名实体识别等功能。 |[WEB](http://ltp.ai/) [Github](https://github.com/HIT-SCIR/ltp)  
基础 | PKUSEG | 北京大学的一款多领域中文分词工具，可以根据不同领域需求进行分词。 |  [WEB]() [Github](https://github.com/lancopku/pkuseg-python)  
基础 | FudanNLP |复旦大学自然语言处理实验室的自然语言处理工具包，提供了分词、词性标注、命名实体识别、依存句法分析等功能。 |  [WEB]() [Github](https://github.com/FudanNLP/fnlp)  
基础 | HanLP | HanLP是一款NLP工具包，提供了中文分词、词性标注、命名实体识别、短语提取、自动摘要等功能。 |[WEB](https://www.hanlp.com/) [Github](https://github.com/hankcs/HanLP)  
基础 | NLPIR |NLPIR是中国科学院计算技术研究所开发的一套自然语言处理工具包，提供了中文分词、词性标注、关键词提取、命名实体识别等功能。通过该工具，可以实现中文依存句法分析。|  [WEB](http://ictclas.nlpir.org/) [Github](https://github.com/NLPIR-team/NLPIR)  
基础 | StanfordNLP |StanfordNLP是斯坦福大学开发的一款自然语言处理工具包，提供了依存句法分析功能。该库是基于Java的StanfordCoreNLP的Python接口，可以处理多种语言，包括中文。 |  [WEB]() [Github](https://github.com/stanfordnlp/stanfordnlp)  
基础 | SpaCy |SpaCy是一个强大的自然语言处理库，提供多种功能，包括依存句法分析。它包含多种预训练模型，可以直接用于中文依存句法分析。 |[WEB](https://spacy.io/) [Github](https://github.com/explosion/spaCy)  
  

## 学习资源库

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
深度学习资源| 动手学深度学习 | 面向中文读者的能运行、可讨论的深度学习教科书 |  [官网](https://zh.d2l.ai/) [Github](https://github.com/ShusenTang/Dive-into-DL-PyTorch)  
编程学习资源| W3Cschool | 一个编程学习网站 | [官网](https://www.w3cschool.cn/html/dict.html)  
基础编程教学| 菜鸟编程 | 提供了编程的基础技术教程,介绍了HTML、CSS、Javascript、Python,Java,Ruby,C,PHP,MySQL等各种编程语言的基础知识。 |[官网](https://m.runoob.com/)  
  
## 深度学习库

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
深度学习框架 | TensorFlow | TensorFlow官方文档。谷歌开源的机器学习框架，适用于深度学习。 |[官网](https://www.tensorflow.org/guide)  
深度学习框架 | PyTorch | PyTorch官方文档。开源的机器学习框架，适用于深度学习。 |[官网](https://pytorch.org/docs/stable/index.html)  
深度学习框架 | Keras | Keras官方文档。高级神经网络API，可用于快速构建深度学习模型。 | [官网](https://keras.io/)  
深度学习框架 | Caffe | Caffe官方文档。用于计算机视觉的深度学习框架。 |[官网](http://caffe.berkeleyvision.org/)  
深度学习框架 | PaddlePaddle | PaddlePaddle官方文档。百度的深度学习框架，支持多种任务。 |[官网](https://www.paddlepaddle.org.cn/documentation/docs/en/2.0-rc1/index_en.html)  
深度学习框架 | MXNet | MXNet官方文档。可扩展的深度学习框架，支持多种编程语言。 |[官网](https://mxnet.apache.org/versions/1.8.0/)  
深度学习框架 | Chainer | Chainer官方文档。Python的深度学习框架，支持动态计算图。 |[官网](https://docs.chainer.org/en/stable/)  
  
## 自然语言处理库

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
Python的自然语言处理库 | NLTK | NLTK官方文档。Python的自然语言处理库，用于文本分析。 |[官网](https://www.nltk.org/)  
Python的自然语言处理库 | SpaCy |SpaCy官方文档。Python的自然语言处理库，用于高级自然语言处理任务，包括词性标注、句法解析和实体识别。 |[官网](https://spacy.io/usage)  
Python的自然语言处理库 | Gensim | Gensim官方文档。Python的自然语言处理库，用于主题建模和文档相似度计算。 |[官网](https://radimrehurek.com/gensim/)  
Python的自然语言处理库 | StanfordNLP |StanfordNLP官方文档。Python的自然语言处理库，由斯坦福大学开发，用于词性标注、命名实体识别、依存句法解析等任务。 |[官网](https://stanfordnlp.github.io/stanfordnlp/)  
Python的自然语言处理库 | TextBlob |TextBlob官方文档。Python的自然语言处理库，用于简化文本处理任务，包括词性标注、名词短语提取、情感分析等。 |[官网](https://textblob.readthedocs.io/en/dev/)  
Python的自然语言处理库 | Pattern |Pattern官方文档。Python的自然语言处理库，用于词性标注、情感分析、数据挖掘等任务，包含一些预处理和可视化工具。 |[官网](https://polyglot.readthedocs.io/en/latest/)  
Python的自然语言处理库 | Flair |Flair官方文档。Python的自然语言处理库，用于词性标注、命名实体识别、情感分析等任务，支持预训练的语言模型和迁移学习。 |[官网](https://github.com/flairNLP/flair)  
Python的自然语言处理库 | AllenNLP |AllenNLP官方文档。Python的自然语言处理库，由AI2开发，基于PyTorch构建，用于构建高级NLP模型，支持多种任务，如语义角色标注、核心引用解析等。| [官网](https://allenai.org/allennlp)  
  
## 机器学习库

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
机器学习库 | Scikit-learn | Scikit-learn官方文档。Python的机器学习库，提供简单高效的工具。 |[官网](https://scikit-learn.org/stable/user_guide.html)  
机器学习库 | XGBoost | XGBoost官方文档。优化的梯度提升机器学习库，用于分类和回归。 |[官网](https://xgboost.readthedocs.io/en/latest/)  
机器学习库 | LightGBM | LightGBM官方文档。高性能的梯度提升机器学习库，用于分类和回归。 |[官网](https://lightgbm.readthedocs.io/en/latest/)  
机器学习库 | CatBoost | CatBoost官方文档。高性能的梯度提升机器学习库，用于分类和回归。 |[官网](https://catboost.ai/docs/)  
  
## 数据分析和操作库

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
数据分析和操作库 | Pandas | Pandas官方文档。Python的数据分析和操作库，提供数据结构和工具。 |[官网](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html)  
数值计算库 | NumPy | NumPy官方文档。Python的数值计算库，提供多维数组和矩阵操作。 |[官网](https://numpy.org/doc/stable/user/index.html)  
Python的绘图库 | Matplotlib | Matplotlib官方文档。Python的绘图库，用于创建静态、动态和交互式图表。 |[官网](https://matplotlib.org/stable/contents.html)  
Matplotlib的数据可视化库 | Seaborn | Seaborn官方文档。基于Matplotlib的数据可视化库，提供美观的图表。 |[官网](https://seaborn.pydata.org/tutorial.html)  
交互式编程环境 | Jupyter | Jupyter官方文档。交互式编程环境，支持多种编程语言。 |[官网](https://jupyter.org/documentation)  
并行计算库 | Dask | Dask官方文档。并行计算库，可扩展Pandas、NumPy和Scikit-learn。 |[官网](https://docs.dask.org/en/latest/)  
  
## 编程语言

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
编程语言 | Python | Python官方文档。通用编程语言，适用于Web开发、数据分析等领域。 |[官网](https://docs.python.org/3/)  
编程语言 | JavaScript | JavaScript官方文档。主要用于Web开发，是主流的前端编程语言。 |[官网](https://developer.mozilla.org/en-US/docs/Web/JavaScript)  
编程语言 | Java | Java官方文档。通用编程语言，常用于企业级应用、Android开发等领域。 |[官网](https://docs.oracle.com/en/java/)  
编程语言 | C++ | C++官方文档。通用编程语言，适用于系统编程、游戏开发等领域。 |[官网](https://en.cppreference.com/)  
编程语言 | C | C官方文档。通用编程语言，广泛应用于操作系统和嵌入式系统开发。 |[官网](https://www.gnu.org/software/libc/manual/)  
编程语言 | Node.js | Node.js官方文档。JavaScript运行时环境，用于开发服务器端应用。 |[官网](https://nodejs.org/en/docs/)  
编程语言 | Ruby | Ruby官方文档。通用编程语言，适用于Web开发、脚本编写等领域。 | [官网](https://www.ruby-lang.org/en/documentation/)  
编程语言 | Go | Go官方文档。通用编程语言，适用于后端开发、分布式系统等领域。 | [官网](https://golang.org/doc/)  
编程语言 | Swift | Swift官方文档。通用编程语言，主要用于苹果平台应用开发。 |[官网](https://docs.swift.org/swift-book/)  
编程语言 | Scala | Scala官方文档。通用编程语言，兼容Java，适用于大数据处理。 | [官网](https://docs.scala-lang.org/)  
编程语言 | Haskell | Haskell官方文档。纯函数式编程语言，适用于学术研究和算法实现。 |[官网](https://www.haskell.org/documentation/)  
编程语言 | R | R官方文档。编程语言和软件环境，主要用于统计计算和数据可视化。 |[官网](https://cran.r-project.org/manuals.html)  
  
## 数据库

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
数据库 | MySQL | MySQL官方文档。流行的开源关系型数据库管理系统，支持SQL。 |[官网](https://dev.mysql.com/doc/)  
数据库 | PostgreSQL | PostgreSQL官方文档。开源的关系型数据库管理系统，支持SQL。 |[官网](https://www.postgresql.org/docs/)  
数据库 | MongoDB | MongoDB官方文档。面向文档的NoSQL数据库，支持JSON格式。 |[官网](https://docs.mongodb.com/manual/)  
数据库 | SQLite | SQLite官方文档。轻量级的嵌入式关系型数据库管理系统，支持SQL。 |[官网](https://www.sqlite.org/docs.html)  
数据库 | InfluxDB | InfluxDB官方文档。时序数据库，适用于大规模数据监控和分析。 |[官网](https://docs.influxdata.com/influxdb/)  
数据库 | Neo4j | Neo4j官方文档。图形数据库，适用于高度关联的数据分析。 | [官网](https://neo4j.com/docs/)  
  
## Web框架

类别 | 资源名（Name） | 描述（Description） | 链接  
---|---|---|---  
Web框架 | Django | Django官方文档。基于Python的Web开发框架，适用于全栈开发。 |[官网](https://docs.djangoproject.com/)  
Web框架 | Flask | Flask官方文档。轻量级Python Web框架，适用于小型项目。 |[官网](https://flask.palletsprojects.com/)  
Web框架 | Express | Express官方文档。基于Node.js的Web开发框架，适用于后端开发。 |[官网](https://expressjs.com/)  
Web框架 | Ruby on Rails | Ruby on Rails官方文档。基于Ruby的Web开发框架，适用于全栈开发。 |[官网](https://guides.rubyonrails.org/)  
Web框架 | Spring | Spring官方文档。基于Java的企业级应用开发框架，适用于后端开发。 |[官网](https://spring.io/guides)  
Web框架 | Vue.js | Vue.js官方文档。轻量级前端框架，用于构建用户界面和单页面应用。 |[官网](https://vuejs.org/v2/guide/)  
Web框架 | React | React官方文档。Facebook开源的前端框架，用于构建用户界面。 |[官网](https://reactjs.org/docs/getting-started.html)  
Web框架 | Angular | Angular官方文档。谷歌开源的前端框架，用于构建动态Web应用。 |[官网](https://angular.io/docs)  
Web框架 | ASP.NET | ASP.NET官方文档。基于.NET的Web开发框架，适用于全栈开发。 |[官网](https://docs.microsoft.com/en-us/aspnet/core/?view=aspnetcore-6.0)  
Web框架 | Laravel | Laravel官方文档。基于PHP的Web开发框架，适用于全栈开发。 |[官网](https://laravel.com/docs)  
  
