import sys,traceback

states=['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X','bogus']

def trim_and_warn(name, max_len, s):
    if len(s) > max_len:
        print("\nWarning - truncating output of %s: your answer has %i characters but the limit is %i" % (
        name, len(s), max_len), file=sys.stderr)
    return s[:max_len]

def check_viterbi(model,n):
  return check_mod_prop(model.get_viterbi_value,n)

def check_mod_prop(getfn,n):
  global states
  try:
    ovflowed=getfn("VERB",n+1) is None
  except:
    ovflowed=True
  statesCheck=True
  for s in states:
    try:
      v=getfn(s,n-1)
    except:
      if s!='bogus':
        statesCheck=False
        break
  return (ovflowed and getfn("VERB",0) is not None and getfn("VERB",n) is not None,
          statesCheck)

def check_bp(model,n):
  return check_mod_prop(model.get_backpointer_value,n)

def a2answers(gdict,errlog):
  globals().update(gdict)
  errs=0
  (cerrs,ans)=carefulBind(
    [('a1aa','model.states'),
     ('a1b','len(model.emission_PD["VERB"].samples()) if type(model.emission_PD)==nltk.probability.ConditionalProbDist else FAILED'),
     ('a1c','-model.elprob("VERB","attack")'),
     ('a1d','model.emission_PD._probdist_factory.__class__.__name__ if model.emission_PD is not None else FAILED'),
     ('a2a','len(model.transition_PD["VERB"].samples()) if type(model.transition_PD)==nltk.probability.ConditionalProbDist else FAILED'),
     ('a2b','-model.tlprob("VERB","DET")'),
     ('a3c', 'model.get_viterbi_value("DET",0)'),
     ('a3d', 'model.get_backpointer_value("DET",1)'),
     ('a4a3','accuracy'),
     ('a4b1','bad_tags'),
     ('a4b2','good_tags'),
     ('a4b3','answer4b'),
     ('a4c','model.get_viterbi_value("VERB",5)'),
     ('a4d','min((model.get_viterbi_value(s,-1) for s in model.states)) if len(model.states)>0 else FAILED'),
     ('a4e','list(ttags)'),
     ("a5t0", "t0_acc"),
     ("a5tk", "tk_acc"),
     ("a5b", "answer5b"),
     ('a6','answer6'),
     ('a7','answer7')
     ],globals(),errlog)
  errs+=cerrs
  s = 'they model the world'.split()
  try:
    ttags = model.tag_sentence(s)
    gdict["viterbi_matrix"] = [[None for _ in model.states] for _ in range(len(s))]
    gdict["bp_matrix"] = [[None for _ in model.states] for _ in range(len(s) - 1)]
    for position in range(len(s)):
        for i, state in enumerate(sorted(model.states)):
            try:
                gdict["viterbi_matrix"][position][i] = model.get_viterbi_value(state, position)
                if position > 0:
                    gdict["bp_matrix"][position-1][i] = model.get_backpointer_value(state, position)
            except NotImplementedError:
                pass
            except Exception as e:
                errs += 1
                print("Exception in computing viterbi/backpointer for %s at step %i:\n%s" % (state, position, repr(e)),
                      file=errlog)
                traceback.print_tb(sys.exc_info()[2], None, errlog)

  except NotImplementedError:
    pass
  except Exception as e:
    errs+=1
    print("Exception in initialising model in adrive2:\n%s"%repr(e),
          file=errlog)
    traceback.print_tb(sys.exc_info()[2],None,errlog)
  (cerrs,nans)=carefulBind(
    [("a4full_vit", "viterbi_matrix"),
      ("a4full_bp", "bp_matrix")],gdict,errlog)
  ans.update(nans)
  errs+=cerrs
  return (ans,errs)

if __name__ == '__main__':
  from autodrive import run, answers, HMM, carefulBind
  with open("userErrs.txt","w") as errlog:
    run(answers,a2answers,errlog)
