from .hit_policy import bestHit, firstHit, excellentHit

def hit(memoryTable:list,memory:int,memory_MAX:int, hit_policy:str )->int:
    

    hit_list = {
        "bestHit" : bestHit.hit, 
        # "firstHit": firstHit.hit,
        "excellentHit": excellentHit.hit
        }
    if hit_policy not in hit_list.keys():
        hit_policy = "bestHit"
    return hit_list[hit_policy](memoryTable=memoryTable,memory=memory,memory_MAX=memory_MAX )
