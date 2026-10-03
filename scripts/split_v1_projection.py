"""Split a V1 ONNX graph around one high precision projection.

Creates FP source graphs only. RKNN conversion and board quality remain separate
validation steps. The isolated graph can use BF16 or INT16 DFP; its two
surrounding graphs retain the remaining V1 mixed precision configuration.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import onnx
from onnx.utils import Extractor


def split(model, gate_output):
    # Deep transformer graphs exceed Python's default recursion depth in
    # ONNX's own Extractor traversal.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    model=onnx.shape_inference.infer_shapes(model)
    graph=model.graph;nodes=list(graph.node)
    # Extractor needs metadata when an original input becomes a cut output,
    # or an original output becomes a cut input.
    typed={v.name for v in graph.value_info}
    for value in [*graph.input,*graph.output]:
        if value.name not in typed:
            graph.value_info.append(value);typed.add(value.name)
    producers={v:i for i,n in enumerate(nodes) for v in n.output if v}
    originals={x.name for x in graph.input};constants={x.name for x in graph.initializer}
    outputs=[x.name for x in graph.output]
    if gate_output not in producers:raise ValueError('Gate output not found')
    index=producers[gate_output];gate=nodes[index]
    if gate.op_type not in ('MatMul','Gemm') or len(gate.output)!=1:
        raise ValueError('Expected one weighted gate projection')
    feature=gate.input[0]

    def ancestors(names, boundaries):
        selected=set();required=set()
        def visit(name):
            if not name or name in constants:return
            if name in boundaries or name in originals:
                required.add(name);return
            if name not in producers:raise ValueError('Unbound tensor '+name)
            i=producers[name]
            if i in selected:return
            selected.add(i)
            for value in nodes[i].input:visit(value)
        for name in names:visit(name)
        return selected,required

    pre_nodes,_=ancestors([feature],set())
    _,weight_runtime_inputs=ancestors(list(gate.input[1:]),set())
    if weight_runtime_inputs:raise ValueError('Gate weights are not static')
    remaining=set(range(len(nodes)))-pre_nodes
    consumed={v for i in remaining for v in nodes[i].input}|set(outputs)
    frontier=[v for i in sorted(pre_nodes) for v in nodes[i].output if v in consumed]
    if feature not in frontier:frontier.append(feature)
    frontier=list(dict.fromkeys(frontier))
    _,pre_inputs=ancestors(frontier,set())
    forwarded=[v for v in outputs if v in frontier or v==gate_output]
    computed=[v for v in outputs if v not in forwarded]
    if not computed:raise ValueError('Post partition has no computed outputs')
    _,post_inputs=ancestors(computed,set(frontier)|{gate_output})
    order=[x.name for x in graph.input]+frontier+[gate_output]
    def ordered(values):return list(dict.fromkeys(n for n in order if n in values))
    extractor=Extractor(model)
    pieces={
      'before_gate':extractor.extract_model(ordered(pre_inputs),frontier),
      'bf16_gate':extractor.extract_model([feature],[gate_output]),
      'after_gate':extractor.extract_model(ordered(post_inputs),computed)}
    for piece in pieces.values():onnx.checker.check_model(piece)
    return pieces,{'gate_node':gate.name,'gate_input':feature,'gate_output':gate_output,
                   'live_frontier':frontier,'original_output_names':outputs,
                   'forwarded_outputs':forwarded,'source_projection_unchanged':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--projection-output','--gate-output',dest='gate_output',required=True)
    p.add_argument('--isolated-format',choices=['bfloat16','w16a16i_dfp'],default='bfloat16')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path,help='Optional named FP inputs for complete split/unsplit ORT parity')
    a=p.parse_args();digest=hashlib.sha256(a.model.read_bytes()).hexdigest()
    if digest!=a.source_sha256:raise ValueError('Source ONNX hash mismatch')
    pieces,report=split(onnx.load(str(a.model)),a.gate_output)
    pieces={'before_projection':pieces['before_gate'],'projection':pieces['bf16_gate'],'after_projection':pieces['after_gate']}
    a.output.mkdir(parents=True,exist_ok=True)
    report.update(scope='FP graph partition only; not quantized or deployed',source_onnx_sha256=digest,parts={})
    for name,model in pieces.items():
        path=a.output/(name+'.onnx');onnx.save(model,str(path))
        report['parts'][name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'inputs':[x.name for x in model.graph.input],'outputs':[x.name for x in model.graph.output],
          'input_dtypes':{x.name:onnx.TensorProto.DataType.Name(x.type.tensor_type.elem_type) for x in model.graph.input},
          'output_dtypes':{x.name:onnx.TensorProto.DataType.Name(x.type.tensor_type.elem_type) for x in model.graph.output},
          'nodes':len(model.graph.node),'requested_format':a.isolated_format if name=='projection' else 'retain_V1_mixed'}
    if a.inputs:
        import numpy as np
        import onnxruntime as ort
        original=ort.InferenceSession(str(a.model),providers=['CPUExecutionProvider'])
        with np.load(a.inputs,allow_pickle=False) as z:values={x.name:z[x.name] for x in original.get_inputs()}
        reference=original.run(None,values)
        for name in pieces:
            session=ort.InferenceSession(str(a.output/(name+'.onnx')),providers=['CPUExecutionProvider'])
            result=session.run(None,{x.name:values[x.name] for x in session.get_inputs()})
            values.update({x.name:y for x,y in zip(session.get_outputs(),result)})
        rows=[]
        for output,ref in zip(original.get_outputs(),reference):
            actual=values[output.name];np.testing.assert_allclose(actual,ref,rtol=1e-4,atol=1e-4)
            delta=np.abs(actual.astype('f8')-ref.astype('f8'));rows.append({'output':output.name,'mae':float(delta.mean()),'max_abs':float(delta.max())})
        report['fp_partition_parity']={'status':'passed','input_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),'outputs':rows}
    (a.output/'split_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
