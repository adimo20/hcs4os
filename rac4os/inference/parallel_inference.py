import dspy
from typing import Any

def parallel_inference(
    module:dspy.Module,
    inputs:list[dict[str, Any]],
    num_threads:int=5,
    **parallel_kwargs   
) -> list:
    """
    Interface for running parallel processing of a classifier module. Use this for cocurrent inference. 
    It takes, the inputs, you would give to the classifier module (RAG, RRR, or agentic approaches),
    as inputs and runs the specified number of threads in parallel. You need to specify all the required
    arguments, like `k` or `where` for RAG or RRR in one input example dict, like in the following example:
    ```
    results = batch_inference(
        rrr,
        [{"query":q, "k":10, "where":{"level":4}} for q in df.name[:10].tolist()],
        batch_size=3    
    )
    ```
    The classifier module must already be initialised before you can insert it here.
    
    Paramters:
        module (dspy.Module): a classifier the package provides, that inherits from dpsy.Module
        inputs (dict[Any]): the inputs the classifier should process
        num_threads (int): number of concurrent instances that process the inputs
        **parallel_kwargs (Any): additional keyword args that will be passed to the dspy paralleliser
    
    Returns:
        list of the processed input in the form dspy prediction objects, following the defined output signature
    
    """
    parallel = dspy.Parallel(
        num_threads=num_threads,
        **parallel_kwargs
    )
    
    exec_pairs = [
        (module, i) for i in inputs
    ]
    
    try:
        results = parallel(exec_pairs)
    except Exception as e:
        raise e 
    
    return results
        