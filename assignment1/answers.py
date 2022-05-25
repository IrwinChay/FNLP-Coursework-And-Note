lm_stats=[5579336, 0.3116466822608252, 0.9853501906482038, 1.4536908944718608e-06, 0.005158178014182952, 0.348077988641086]
best10_ents=[(2.4921691054394848, ['and', 'here', 'is', 'proof', 'the']), (2.5390025889056123, ['and', 'bailed', 'he', 'here', 'is', 'man', 'on', 'that', 'the']), (2.5584079236733106, ['is', 'the', 'this', 'weather', 'worst']), (2.5686534278173125, ['s', 's', 's', 's', 's', 's', 's', 's', 's', 's']), (2.569853705187651, ['be', 'bus', 'here', 'the', 'to', 'want']), (2.576919752608039, ['hell', 'that', 'the', 'was', 'wat']), (2.587767243678531, ['creation', 'is', 'of', 'on', 'story', 'the', 'the']), (2.5885860368906832, ['fro', 'one', 'the', 'the', 'with']), (2.595298329492654, ['is', 'money', 'motive', 'the', 'the']), (2.617870705175611, ['at', 'bucks', 'end', 'lead', 'of', 'the', 'the', 'the'])]
worst10_ents=[(17.523736748003564, ['作品によっては怪人でありながらヒーロー', 'あるいはその逆', 'というシチュエーションも多々ありますが', 'そうした事がやれるのもやはり怪人とヒーローと言うカテゴリが完成しているからだと思うんですよね', 'あれだけのバリエーションがありながららしさを失わないデザインにはまさに感服です']), (17.524868750262904, ['ロンブーの淳さんはスピリチュアルスポット', 'セドナーで瞑想を実践してた', 'これらは偶然ではなく必然的に起こっている', '自然は全て絶好のタイミングで教えてくれている', 'そして今が今年最大の大改革時期だ']), (17.5264931699585, ['実物経済と金融との乖離を際限なく広げる', 'レバレッジが金融で儲けるコツだと', 'まるで正義のように叫ぶ連中が多いけど', 'これほど不健全な金融常識はないと思う', '連中は不健全と知りながら', '他の奴がやるから出し抜かれる前に出し抜くのが道理と言わんばかりに群がる']), (17.527615646393077, ['一応ワンセット揃えてみたんだけど', 'イマイチ効果を感じないのよね', 'それよりはオーラソーマとか', '肉体に直接働きかけるタイプのアプローチの方が効き目を感じ取りやすい', '波動系ならバッチよりはホメオパシーの方がわかりやすい']), (17.53293217459052, ['慶喜ほどの人でさえこうなんだから', '並の人間だったらなおさら参謀無しじゃ何も出来ない', '一般に吹聴されてる慶喜のネガティブ論は', 'こうした敵対勢力による相次ぐテロに対して終始無関心で', '慶喜個人だけに批判を向けがち']), (17.541019489814225, ['昨日のセミナーではお目にかかれて光栄でした', '楽しく充実した時間をありがとうございました', '親しみのもてる分かりやすい講演に勇気を頂きました', '素晴らしいお仕事とともに益々のご活躍願っております', '今後ともよろしくお願いします']), (17.541411086467402, ['自民党が小沢やめろというなら', '当然町村やめろというブーメランがかえってくるわけです', 'おふたりとも選挙で選ばれた正当な国民の代表ですから', 'できればどちらにもやめてほしくありません', 'そろそろこんな不毛なことはやめにしてほしい']), (17.5427257173663, ['知識欲というのは不随意筋でできている', 'どうせ人間には永久に解明できないんだから', '宇宙はある時点で生まれたのか', 'それとも永遠の過去から存在しているのかなんてことを追究するなと言ってもムダだ', '心臓に止まれと命令しても止まらないのと同じことだ']), (17.547644050965395, ['と言いつつもやっぱり笑えない時はあるよなあ', '笑っても自分の笑顔が汚らわしく思えてすぐ止めちゃうの', '自分が息してるだけで悲しくてぼろぼろ泣いてる時期もあった', '今の自分に必要な経験だったとは思うけど', '出来ればあんな感情は二度とごめんだ']), (17.55280652132174, ['中身の羽毛は精製過程で殺菌処理しているから', '羽毛布団からダニが湧くことはない', 'あと羽毛布団の生地は糸の打ち込み本数が多く', '羽毛の吹き出しを防ぐ目つぶし加工をしているからダニは羽毛ふとんの生地を通過できない', 'ただダニが布団に付着することはあるから手入れは必要'])]
answer_open_question_3='The beginning of the lists have correctly spelled, short, and common English words (e.g. and, is, the) \nwith lower entropy (2.5). The end of the lists have long, rare in English, and non-latin characters with \nhigh entropy (17.5). The end of the lists are being assigned a lower certainty since they are being \nconsidered as unseen data by the bigram model based on Brown corpus which contains English words only. '
answer_open_question_4='Problem: data contains many misspelled words since Twitter users do not always follow formal English spellings;  \nTechnique: filter out; or apply spelling correction by edit distance on the data; \n\nProblem: data contains many word forms, abbreviation, or slang (e.g. FNLP, IAML)\nTechnique: filter non-formal words out, or apply clustering by a word embedding (or simply lemmatization) to combine different slangs with similar \nmeanings if they provide important information to the task;'
mean=3.8435755769050926
std=0.47772976561662
best10_ascci_ents=[(2.4921691054394848, ['and', 'here', 'is', 'proof', 'the']), (2.5390025889056123, ['and', 'bailed', 'he', 'here', 'is', 'man', 'on', 'that', 'the']), (2.5584079236733106, ['is', 'the', 'this', 'weather', 'worst']), (2.5686534278173125, ['s', 's', 's', 's', 's', 's', 's', 's', 's', 's']), (2.569853705187651, ['be', 'bus', 'here', 'the', 'to', 'want']), (2.576919752608039, ['hell', 'that', 'the', 'was', 'wat']), (2.587767243678531, ['creation', 'is', 'of', 'on', 'story', 'the', 'the']), (2.5885860368906832, ['fro', 'one', 'the', 'the', 'with']), (2.595298329492654, ['is', 'money', 'motive', 'the', 'the']), (2.617870705175611, ['at', 'bucks', 'end', 'lead', 'of', 'the', 'the', 'the'])]
worst10_ascci_ents=[(5.166314124571327, ['hoje', 'nossa', 'amiga', 'espero', 'q', 'sorte', 'tenha', 'vc']), (5.166378486661663, ['aok', 'berlin', 'brandenburg', 'bürofläche', 'commercial', 'engel', 'immobilien', 'meldung', 'mietet', 'potsdam', 'potsdam', 'qm', 'v', 'völkers']), (5.166607294898407, ['mi', 'rt', 'ixxi', 'squeciduu', 'yinha']), (5.166636591109558, ['asdhiasdhiuadshiuads', 'rt', 'tentaando', 'to', 'x']), (5.166680391667252, ['aaaaaai', 'aqui', 'e', 'é', 'gente', 'guri', 'horror', 'lente', 'não', 'não', 'o', 'olho', 'que', 'que', 's', 'tem', 'tem', 'um', 'vermelho']), (5.166696700850563, ['aaaon', 'aah', 'as', 'cow', 'da', 'eu', 'lindas', 'parade', 'vaquinhas', 'viii']), (5.16672881298096, ['bra', 'di', 'douglas', 'douglas', 'ett', 'finansmannen', 'för', 'för', 'gustaf', 'gustaf', 'konkurrensen', 'och', 'skogsägaren', 'slag', 'slår', 'sveaskog']), (5.166820287702661, ['bola', 'macaé', 'rolando', 'vasco', 'x']), (5.1670133043123725, ['enaknya', 'hari', 'hmmm', 'ini', 'kmn', 'ya']), (5.16719955611981, ['ad', 'emg', 'ha', 'haha', 'ak', 'jg', 'k', 'kreta', 'krta', 'ksmg', 'mau', 'mba', 'naik', 'rt', 'smg', 'wkwk'])]
best10_non_eng_ents=[(4.321320405509358, ['afganistán', 'asociación', 'de', 'de', 'mujeres', 'rawa', 'revolucionarias']), (4.321322108677053, ['carrey', 'face', 'feat', 'mariah', 'minaj', 'my', 'nicki', 'out', 'rt', 'up', 'video', 'xxlmag', 'com']), (4.321338322311484, ['abisss', 'aja', 'demo', 'gini', 'hari', 'hikmah', 'mantabsss', 'membawa', 'sepi', 'sudirman', 'thamrin', 'tiap', 'trnyta']), (4.321374586541906, ['a', 'a', 'agora', 'com', 'consegui', 'd', 'de', 'dormir', 'dormir', 'durmo', 'e', 'eu', 'inteira', 'mas', 'nao', 'nao', 'nao', 'nao', 'noite', 'noite', 'nove', 'q', 'se', 'sono', 'to', 'vou']), (4.321385367081537, ['eh', 'meu', 'o', 'que', 'twitter', 'esse']), (4.321390995790845, ['don', 't', 'get', 'giggle', 'i', 'i', 'oh', 'oh']), (4.321500139376771, ['am', 'geburtstag', 'gefeiert', 'klingende', 'november', 'töne', 'wird']), (4.321525080235015, ['attraktion', 'belønning', 'fest', 'kærlighed', 'lykke', 'privacy', 'succes', 'tarot', 'tillid', 'udholdenhed']), (4.321575437028815, ['cerca', 'de', 'de', 'deci', 'del', 'exameeen', 'examen', 'le', 'mateeee', 'matematica', 'pueda', 'que', 'rt', 'shhuu', 'shuuu', 'sientese', 'valla', 'wayyy', 'y']), (4.321622845063304, ['abre', 'china', 'huaehuiaieh', 'o', 'olho'])]
worst10_non_eng_ents=[(5.166314124571327, ['hoje', 'nossa', 'amiga', 'espero', 'q', 'sorte', 'tenha', 'vc']), (5.166378486661663, ['aok', 'berlin', 'brandenburg', 'bürofläche', 'commercial', 'engel', 'immobilien', 'meldung', 'mietet', 'potsdam', 'potsdam', 'qm', 'v', 'völkers']), (5.166607294898407, ['mi', 'rt', 'ixxi', 'squeciduu', 'yinha']), (5.166636591109558, ['asdhiasdhiuadshiuads', 'rt', 'tentaando', 'to', 'x']), (5.166680391667252, ['aaaaaai', 'aqui', 'e', 'é', 'gente', 'guri', 'horror', 'lente', 'não', 'não', 'o', 'olho', 'que', 'que', 's', 'tem', 'tem', 'um', 'vermelho']), (5.166696700850563, ['aaaon', 'aah', 'as', 'cow', 'da', 'eu', 'lindas', 'parade', 'vaquinhas', 'viii']), (5.16672881298096, ['bra', 'di', 'douglas', 'douglas', 'ett', 'finansmannen', 'för', 'för', 'gustaf', 'gustaf', 'konkurrensen', 'och', 'skogsägaren', 'slag', 'slår', 'sveaskog']), (5.166820287702661, ['bola', 'macaé', 'rolando', 'vasco', 'x']), (5.1670133043123725, ['enaknya', 'hari', 'hmmm', 'ini', 'kmn', 'ya']), (5.16719955611981, ['ad', 'emg', 'ha', 'haha', 'ak', 'jg', 'k', 'kreta', 'krta', 'ksmg', 'mau', 'mba', 'naik', 'rt', 'smg', 'wkwk'])]
answer_open_question_6='Sparse data problem: since zero probability exists for possible sequence, the corpus can never represent all English language. Independence assumption: P(word) only depends on a fixed number of history\n\nCorpus problem: with similar words, some use of language is more predictable. \nAssumption: corpus used contains all words and all form of English, development set drawn from same source as training set \n\nModel problem: only cross entropy could be measured instead of actual entropy, and different models shows different performance. \nAssumption: the model used could compress the data with highest efficiency and its cross entropy = entropy \n\nSince per word cross entropy could be approximated by the average negative log probability a model assigns to each word, a Ngram model (with smoothing such as back off) is trained by MLE to estimate probability of next word. The model is then tested on another development set. As N increases, the cross entropy approaches the entropy of English.'
naive_bayes_vocab_size=13521
naive_bayes_prior={'N': 0.5223306571799433, 'V': 0.47766934282005674}
naive_bayes_likelihood=[0.006913064743369809, 0.0012190937826217086, 0.12333945519178972, 2.2315401420598457e-06, 2.6766530157362866e-05, 0.004917741586184577, 0.004933935254094318]
naive_bayes_posterior=[{'N': 0.41139627956588926, 'V': 0.5886037204341108}, {'N': 0.8436673290620547, 'V': 0.1563326709379453}, {'N': 0.18757809621627583, 'V': 0.8124219037837241}, {'N': 0.18757809621627583, 'V': 0.8124219037837241}, {'N': 0.0038562513284388163, 'V': 0.9961437486715612}]
naive_bayes_classify=['V', 'N']
naive_bayes_acc=0.7949987620698192
answer_open_question_8='The accuracy of only using feature P is higher: it provides the greatest information. So the attachment of prepositional phrases is mostly determined by the choice of preposition word. All 4 words could not be considered redundant stopwords, single used or combined. The label (N1, N2) also lowers uncertainty. \n\nThe accuracy of Q7 is 79.5%. So dependences between features lower some uncertainty. Naive Bayes also assumes all features equally important but P provides more information.  '
lr_predictions='VVVVVVVVVNVVVVVVVVVVVVVVVVNVVNVVVVVVVVNVVVVVNVNNNNVVNVVNNNVNNNVNNVVVNNVVVNVNVVNVNVNVNNNVVVNVNNNVNVNVNNVNVVVVNVNVVNNNNVVVNVVNVNVVNNVNNVNVNNNVVNVNNVNNNNVNNNNNVNNNNVNNNNVNNNNNNNVNVVVVVVVNVNVNNNNVVVVNVVVVVVNNNNNVVNVNVNNNNVNVVVNVNVNNNNVNVNVNNVNNVNNVNNVNVVNNNNVNVNVNNVNVVNNNNVVNVNVNVVNVVVNVVNVNNVNNVVNNNVNNVVNNNNNVNVNNVVVNNNVVVVNVVVVVVVNNNNVVVNNNNNVNVNVNNVVNVNNNNNVNVNVVNVVVNNNVVNNVVVVVVNNNNNNVVNVVNNNNNNVNNVVVNNVVVVVVVVNNNVNVNVVNNNVVVVVVVVVNVVVVNVNVNNVVNVVVNVNNNNNVVNVNNNVNVNNNVVNVNNVNNNVNNNVNVNVVVVVVNVVNVNNVNNVNNVVVNVNNVVVVVVVVVVVVVVVVVNVVVNVNNVVNVVNNNNVNVNNNVNVNVNVVNNNNNNVVNNNNVVVVVNNVNVNNNNVNVVVVNVVNVNVVVVVVNNVNNNVVNNVVVNNNNVNVVNVVNVNVVVNVVVVNVVVNVNVVVVNNVNVVNVVNVVNNNNVVNNVNVNVVNNVNVVVVNVVVVVVVVVVVVVVVVNVNVVNVVVVVVVNVNNNNNVVNNVNVNNNNVVVVNVVVVNVNVVNVVVNVNNNNNNNVNNNNVNNVVVNNNNVNNVVVNVVVVVVVVNVNNNVVVNVNVVNVNNNVVNVNNNVNVNVNVVNVVNVVNNNNNNNVVNVVVNVVVVNVNVVNVNNNNVNNNNNVNVNNNNVVNNVNNVVVVVNVVNNNVNNNNVNVVVNVNNNVVNVVNNNVNNVNNNNNNVNNNVNVVVNVVVNVVNNVNNNVVNVVVVNVNNNNNVVVNNNVVNVNNNNNNNVVVVNVVNVVVVVNVVVVVVNNVVVNNNVVVNNVNNVNNNNNNNVVVNNNNVVVVVNVNVNVVVNNVNNVNVVNNNVNVNVNNVNNNVNVVVVVNVVNVNNVVVNNNNNVNNNNVNVNNVNNNNNNVVNNVVNNNNNNNVVVVNVNNNVNVNNNVVVVVVVVNNVVVNVNNVNVVVVNNVVNVVNVVVNVVVVVNNNNNNNNNNNNNNVNNVNNNVNNVNNVVVVNVVNVNNNNVVNVVVNNVNNNVVVNVNNVVNNNVNNNVNNNNVVVNVVNVNNNNVNNVNNVVNVVVNVNVNNNNVVNNNNNVVNNVNNNNNNVVVVNNVVVNVNVNNVVVNVNNVVVNVVNVVNVNNVNVNVNVVNNVVVNVVVVVNVNNNVNNNNVNVVNNNVVNNVNNVNVNNVVVNVNNNVVVNNVVVVNNNNVNVNNVVVVVNNNNNNVNVVNVNVVVVNNVNNNNVVNVVVNNVNVVNNVNNNNNVNNVNNNVNNNNNNVNVNNNNVVVNVVNNVNNVVNNNNNNVNNVVNNVVVVNNVNNNVVVNNNNVNNVNVVNNVNNVNNVNNNVNVNNVNNVNNNNVVNVNVVNNVNVNVNVNNVVVVVVNNVNNNNNVVVVNNNVVVVNVNVNVNVNNVNNNNNNVNVNVNNVVNNVNNNVNVVVVNVNNNVNVNVNVVVNVVNVVNVVNNVVVNVNNVVNNVVNNNVNVNNNVVNVNNVNVNVVNNVNNVNVVNVVVNNNVNNNVNNNNNNNNNNNNNVVVNVVVNVVVNNVNVVNVNVVVNNVVNNNNVVNVNVNVNVNVNVVVNVNNNNVVNVNNVNNVNNNNNNNVNVNVNNVNNNVVVVVVVNNNNNVNVNVNVNVNNVVNNNNNVNNNNNVNNVNVVNVNNVNNNVVVNVNVNVVNNVNVNVVVVVNNNVNNNNNVNVNNNNVNVVNNNNNNVNVNVVVVNVVNNNNVNNVNVVNNNNNVVNNVNVNNNNVNNVVNNVNNNNVNNVNVNVVVVNVNVNVNNVNVNVVVVVNVNVNVNVNNNVVVVNVNVNNVVNNNVVNNVNNVNNVNVVNNVNVNNNNVVVNVVVNNNVVVNNNVVVVVNVVVVNVNVVNVVNNVVVNVVNNVNVVVNVVVNVNVVNNVVVNNVVNNVVVVNNNNNVNVVNNNNNNVNVVVNVVNVNVVNNNVNNVNVNVVNNVVVVNNNVVVNNVNVVNNVNNVVVNVNVNVNVNNVVVVVVVVNNNVNVNVVVNVNNNNNNVVVNNNNVNNVVNVVVNVNNVVNVVNVNNVNVVVNNNVNVNVNVNNVVVVVNNNVVVVVNVVNVVVNNNNVVNVVNNVVVNNNVNNNNVNVNNVVNVNNVVVNNNNVNVVNVVNVNVNNVVVVNNVNVNVVVVVVNNNVVNVNNNNNVVNNVNNNVNNVNNVNVVVNVVVNNVVNNNNNVVVNNVVNNNNVVNVVNNVNVNNNNNVVVVVNNNVNNVVNVVVVVNVNVVVVVVVNNNVNNNVNNVVVVNVNVNVNNVVVNNNVNNNNVNNVVNVVNVNNNNVVVNVVNVNVNNVVNVNVVNVNVVVVVNVNVVVNNVVVVNNNVVVVNNVVVNNNVVVNNVVVNNNNNVVVNVNVVVNNVNNVVVVVVNNNNNNVVVVVVVVVNNVNVVNNNNNNVVVNVVVNNNNVVVNVNNVNNNNNVVVNVVNVVNVNNVVVVVVNVNVVVVVVVVNVNNNNNVVNVNVNVVNNVVNVNNVNVNNNNNVVVNNNVVVVVNNVNNVVVNNVNVVVVVVNNVVVVVNNVNVNNNNNVNNNVNNVVVVVNVVVNNVVVVNNNNVNNNVVVNVVVVVVVVNNVNVNVVNNVNVVVVVVNNVVVNVVVNVVVVNVVVNNVNVNNVVVNVNVVNNNVNVNNVVNNNVVNNVNNVNVNVNNVNVNNNNNVNNNVVVVNNNNNNNNNVNVNNVNVNNNNNVNVVNNNNVVVVVNVNNNVVVNVNVVNVVNNNNNNNVVVVVVNVNVVVNVVVVVNNVVNNVNNNVVVVNNNNNVVNVVNVVVNNNNVVNNVNNVVVVVVVNVVVNVNVNVNVNVVNNNVNNNVNVNNNNNNVVVNVVVNNVNNVNNNVNVNNNNNNNVVVVNVNVNVVVVVVVVVVVNVVVVVNNVVNNVVNVVVVNNVNNNNVVVVNNNNVVVVVVNNNNNVNVNNNVNNNVVNVNVVVVVNNVNVNNNNNVVNNNNVVVVVVVVVNVVVVNVVNNNNVNNNNNVNVVVNNVNNNVVVNNVVNNNVNNVVVNVNNVNNNVVNNNVVVVNVVNNVNNNVVVVVNVNVNNVVNNNVNNVNNVVVVNNVNVNNNVNVNVNNVVVVNVNVNNVNNVNNNVVVNNVNNVVVVVVNVNVVNVVVNVNVVVVNNVVVNNVVNNNNVVNNNVNVVNNNVVVNVNNNNVNNNVVVVVVVVVVNVVVNNNNNVVVNVVNNNVNVNVNNNNNVVNVVNNNVVVNVVVNNNVVVNVVNNNVNNVVVVNNVNNVNVVNNVNNNNVNVVNNNNVVVNNNVVNVVNNNNVVVVVVVNVVNNVVNNNNVVVVVVNNVVVVNNVNNVNVNNNVNVVNVVNNNNVVNVVNVNVVVNVNNNVVVVVVNVVVNNVNVNNNNNNVNVVNVVNVVVVVVNNNNNVVNNNNVVNVVVNNNVNNNNNNVVVNNNVVNNVVNVVNVVVNNVNNVVVVNNVVNNVNNVVVVVNVNNVVVNNNVNNVVNNVNVVVNNNNVVNNVNNVNVNVVVVNNNNVNVVNNNNNVNNVVNVVVNNNVVNNNNVVNNNVNVNVNVNVNVVVVVNNNNVVNVNNNVNNNVNNNNNNNVNVVVVNNNVNNNNNNNNVVNVNVVNVVNVNNNVNNNNNNNNNNVNNNVNVNVNVNVNVNNVNVNVNVNVNVVVNNNVNVNNNNNVNNNNNNVNVVVVNNVNVNNNNNNVNNVNNVVNNVNVVVVVNVVNNNVVNVVVNNVVNVNVNVNVVNVVNNVNVVVVVVNNNNNNNNNVNVNNVVNNNNNNNNVNVNNNVVVVNVNVNVNVNVNVVVNNNNNNVVVVNNNNNNNNVNNVVVNVVVNVNVNVVNNVVVVNNVNVNVNVNNVVNVNNVVVVVVNVVNNVNVNNVN'
answer_open_question_9="Since the vocab contains different forms of the same word (e.g. companies) but they are encoded independently, lemmatization is used to cluster them. When applying only to N1, lemmatization contributes the greatest improvements (0.3%). \n\nSince feature P provides the most information, I concatenate it with other single features to further emphasize the use of P. (3% acc) \n\nE.G: The feature ('p', 'of') and features containing P (e.g. v+p) have the highest weights since the model depends on the choice of preposition to determine the attachment. \n\nSince sequential features might lower the model’s uncertainty, I encoded the features as uni, bi & trigram and concatenated them to resemble interpolation. (0.8% acc) \n\nE.G: So trigram features have some of the highest weights (2.49, 2.40) \n\nE.G: the feature '1988' is an outlier since it does not provide information to disambiguate PP. By inspection, all 3 occurrences belong to the ‘V’ class. It might be a bias in corpus captured by the model "
