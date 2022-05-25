###!!!! This version works when embedded in a template file, see assignment2/template.py !!!!
### For separated use, see autodrive_new.py
import sys, re, importlib, traceback, os, inspect, ast
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
  with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:  
      yield
    finally:
      sys.stdout = old_stdout

if False:
  # This has to be _outside_ the function below,
  #  so the e.g. answer function in the input module is defined
  msg=None
  if len(sys.argv)==1:
    msg='Solution filename missing'
  elif len(sys.argv)>2:
    msg='No arguments allowed, just a file name'

  if msg is None:
    modFilename=sys.argv[1]
    mres=re.match("([^.]*)\.py$",modFilename)
    if mres is None:
      msg='%s is not a valid python filename'%modFilename

  if msg is not None:
    print("""%s
    Usage (in fnlp environment): python autodrive.py [solution].py"""%msg,file=sys.stderr)
    exit(1)

  modName=mres.group(1)

  try:
    mod=importlib.import_module(modName)
    # do, as it were, from modName import *
    attrs = { key: value for key, value in mod.__dict__.items() if
              key[0] != "_" }
    globals().update(attrs)
  except (ModuleNotFoundError, ImportError) as e:
    print("Filename %s must be importable: %s"%(modFilename,e),file=sys.stderr)
    print("Search path was: %s"%sys.path,file=sys.stderr)
    exit(2)

FAILED='failed'
errs=0

def safeEval(expr,gdict,errlog):
  global FAILED,errs
  try:
    return eval(expr,gdict)
  except NotImplementedError as e:
    # Will have been logged already
    errs+=1
    return FAILED
  except Exception as e:
    errs+=1
    print("""Exception in answer dict value computation:
    %s -> %s"""%(expr,repr(e)),file=errlog)
    traceback.print_tb(sys.exc_info()[2],None,errlog)
    return FAILED    

def carefulBind(aitems,gdict,errlog):
  global errs
  errs=0
  return (errs,{ k:safeEval(v,gdict,errlog) for k,v in aitems })

def run(gdict,answer,answerFactory,errlog,grabPlots=False):
  global counter
  # Override plot to force save to png
  # Thanks to http://stackoverflow.com/a/28787356
  if grabPlots:
    from matplotlib import pylab, pyplot

    counter = 0
    def filename_generator():
      global counter
      counter += 1
      return 'plot_%s.png'%counter

    def my_show(**kwargs):
      res=pylab.savefig(filename_generator())
      return res

    pylab.show=my_show
    pyplot.show=my_show

  sys.path=['.']+sys.path

  indent=re.compile('\s\s*')
  ncs=re.compile('[^"\']*["\'][^"\']*#')
  cclean=re.compile('\s*#.*$')
  mstart=re.compile('(\s*).*[:\\\\,(%]$')
  prefixTemplate='\s{%s,%s}[^\s].*[^:]$' #.*[^:\\\\]$
  candef=re.compile('(\(?[,\sa-zA-Z_0-9]*\)?)=')
  globals().update(gdict)

  with suppress_stdout():
    print("Starting run, please be patient",file=sys.stderr)
    sys.stderr.flush()
    aLines=inspect.getsource(answer).split('\n')[1:]
    errs=0
    multi=None
    prefix=None
    ii=None
    inLong=False
    triples=['"""',"'''"]
    mbs=False
    for a in aLines:
      if ii is None:
        al=indent.match(a)
        if al is None:
          continue
        else:
          ii=len(al.group(0))
      if len(a)>0 and ('#' not in a[:ii]):
        a=a[ii:]
      if inLong:
        if a[-3:] in triples:
          inLong=False
        continue
      if a[:3] in triples:
        inLong=True
        continue
      if not ncs.match(a):
        a=cclean.sub('',a)
      if len(a)==0:
        continue
      #print('dbg',a,mbs,file=sys.stderr)
      if (multi is not None) and (not mbs) and prefix.match(a):
        #print('eom',a,file=sys.stderr)
        # we're done with the multi-line, so do it
        #print("Multi:",multi,file=sys.stderr)
        try:
          exec(multi,gdict)
        except Exception as e:
          errs+=1
          bogus=(" %s:"%e.args[0]) if len(e.args)>0 else ''
          print("""The following lines threw a %s exception:%s
%s
-------"""%(e.__class__.__name__,bogus,multi),
          file=errlog)
        multi=None
      if multi is None:
        m=mstart.match(a)
        if m is not None:
          pl=len(m.group(1))
          prefix=re.compile(prefixTemplate%(pl,pl))
          multi=a
          mbs=multi[-1] in ['\\',',','(','%']
          #print('mstart, |%s|, mbs=%s, pfx=|%s|'%(a,mbs,prefix.pattern),file=sys.stderr)
          continue
        else:
          pass # fall through to treat this as a single line expression
      else:
        if multi[-1] in ['\\',',','(','%']:
          multi=(multi[:-1] if multi[-1]=='\\' else multi)+a
          mbs=multi[-1] in ['\\',',','(','%']
        else:
          #print('mml','|%s|'%a,mbs,multi,file=sys.stderr)
          multi=multi+'\n'+a
        continue
      defaulted=False
      try:
        #print('execa',a,file=sys.stderr)
        exec(a,gdict)
      except Exception as e:
        errs+=1
        if len(e.args)>0:
          if isinstance(e.args[0],tuple):
            (dm,dvv)=e.args[0]
            maydef=candef.match(a)
            if maydef:
              dv=ast.literal_eval(dvv)
              bmsg="%s, defaulted to %s"%(dm,dvv)
              exec("%s=%s"%(maydef[1],repr(dv) if isinstance(dv,str) else dv),gdict)
              defaulted=True
            else:
              bmsg="%s, couldn't default???"%dm
          else:
            bmsg=e.args[0]
          bogus=" %s:"%bmsg
        else:
          bogus=''
        print("""The following line threw a %s exception:%s
%s
-------"""%(e.__class__.__name__,bogus,a),
        file=errlog)
  try:
    gdict.update({('FAILED',FAILED)})
    (ansd,uerrs)=answerFactory(gdict,errlog)
    errs+=uerrs
  except Exception as e:
    print("""Failed to compute answer dict:
    %s
    """%e,file=sys.stderr)
    traceback.print_tb(sys.exc_info()[2],None,sys.stderr)
  # dump for automarker
  with open("answers.py","w") as f:
    for aname,aval in ansd.items():
      if aval is FAILED:
        errs+=1
        vstr=''
      else:
        vstr=repr(aval) if isinstance(aval,str) else aval
      print("%s=%s"%(aname,vstr),file=f)
  if errs==0:
    os.remove('userErrs.txt')
  else:
    print("%s errors caught during answer processing, see userErrs.txt"%errs,file=sys.stderr)
