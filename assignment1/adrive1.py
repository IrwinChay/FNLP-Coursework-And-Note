from autodrive_embed import carefulBind

def extract_answers(gdict, errlog):
  globals().update(gdict)
  (cerrs,ans)=carefulBind(
        [('lm_stats', '''[lm._N,
                             lm.prob('h', 't'),
                             lm.prob('u', 'q'),
                             lm.prob('z', 'q'),
                             lm.prob('j', ('<s>',), True),
                             lm.prob('</s>', 'e', True)]'''),
        ('best10_ents', "best10_ents"),
        ('worst10_ents', "worst10_ents"),
        ('answer_open_question_3', "answer_open_question_3"),
        ('answer_open_question_4', "answer_open_question_4"),
        ('mean', 'mean'),
        ('std', "std"),
        ('best10_ascci_ents', 'best10_ascci_ents'),
        ('worst10_ascci_ents', 'worst10_ascci_ents'),
        ('best10_non_eng_ents', 'best10_non_eng_ents'),
        ('worst10_non_eng_ents', 'worst10_non_eng_ents'),
        ('answer_open_question_6', "answer_open_question_6"),
        ("naive_bayes_vocab_size", "len(naive_bayes.vocab)"),
        ("naive_bayes_prior", "naive_bayes.prior"),
        ("naive_bayes_likelihood", '''[naive_bayes.likelihood["V"][("v", "rose")],
                     naive_bayes.likelihood["V"][("p", "of")],
                     naive_bayes.likelihood["N"][("p", "of")],
                     naive_bayes.likelihood["N"][("n2", "609")],
                     naive_bayes.likelihood["V"][("n2", "609")],
                     naive_bayes.likelihood["V"][("n1", "million")],
                     naive_bayes.likelihood["N"][("n1", "million")]
                     ]'''),
        ("naive_bayes_posterior", '''[naive_bayes.prob_classify([("v", "took")]),
                                  naive_bayes.prob_classify([("v", "took"), ("n1", "advantage")]),
                                  naive_bayes.prob_classify([("p", "to")]),
                                  naive_bayes.prob_classify([("n1", "dsfjghdkfjgh"), ("p", "to")]),
                                  naive_bayes.prob_classify(
                                      [("v", "values"), ("n1", "plan"), ("p", "at"), ("n2", "billion")])
                                  ]'''),
        ("naive_bayes_classify", '''[naive_bayes.classify([("v", "took")]),
                              naive_bayes.classify([("v", "took"), ("n1", "advantage")])]'''),
        ("naive_bayes_acc", "naive_bayes_acc"),
        ('answer_open_question_8', "answer_open_question_8"),
        ("lr_predictions", '"".join(logistic_regression_model.classify(d) for (d, gold) in dev_features)'),
        ('answer_open_question_9', "answer_open_question_9"),
     ], globals(), errlog)

  return ans, cerrs
