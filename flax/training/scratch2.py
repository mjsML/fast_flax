from concurrent.futures import thread

def save_checkpoint(bla,blo,bli,blu=19,blj=True):
    _kwargs= locals()
    def _save_checkpoint(bla,blo,bli,blu=19,blj=False):
        _kwargs= locals()
        return _kwargs
    if blj:
        with thread.ThreadPoolExecutor(max_workers=1) as executor:
            
            future = executor.submit(_save_checkpoint, **_kwargs)
            return future
    return _save_checkpoint(**_kwargs)
hs=[]
h=save_checkpoint(1,2,3,4)
hs.append(h)
h=save_checkpoint(1,2,3,4)
hs.append(h)
h=save_checkpoint(1,2,3,4)
hs.append(h)

print([(r.result(),type(r)) for r in hs])

