
import time
import functools

def with_retry(max_attempts=3,base_delay=1.0,exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args,**kwargs):
            last_exc=None
            for attempt in range(1,max_attempts+1):
                try:
                    return fn(*args,**kwargs)
                except exceptions as e:
                    last_exc=e
                    wait=base_delay * (2 ** (attempt-1))
                    print(f"[Retry] {fn.__name} failed (attempt {attempt}/{max_attempts}) : {e}. Retrying in {wait}s")
                    if attempt < max_attempts:
                        time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator