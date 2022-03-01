# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import numpy as np
import json, os, math
from collections import defaultdict

"""
Utilities for working with function program representations of referring expressions.

Some of the metadata about what referring expressions node types are available etc are stored
in a JSON metadata file.
"""


# Handlers for answering referring expressions. Each handler receives the scene 
# structure that was output from Blender, the node, and a list of values that 
# were output from each of the node's inputs; the handler should return the 
# computed output value from this node.


def scene_handler(scene_struct, inputs, side_inputs):
  # Just return all objects in the scene
  return list(range(len(scene_struct['objects'])))


def make_filter_handler(attribute):
  def filter_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    value = side_inputs[0]
    output = []
    for idx in inputs[0]:
      atr = scene_struct['objects'][idx][attribute]
      if value == atr or value in atr:
        output.append(idx)
    return output
  return filter_handler


def unique_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  if len(inputs[0]) != 1:
    return '__INVALID__'
  return inputs[0][0]


def vg_relate_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  assert len(side_inputs) == 1
  output = set()
  for rel in scene_struct['relationships']:
    if rel['predicate'] == side_inputs[0] and rel['subject_idx'] == inputs[0]:
      output.add(rel['object_idx'])
  return sorted(list(output))


def relate_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  assert len(side_inputs) == 1
  relation = side_inputs[0]
  return scene_struct['relationships'][relation][inputs[0]]
    

def union_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return sorted(list(set(inputs[0]) | set(inputs[1])))


def intersect_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return sorted(list(set(inputs[0]) & set(inputs[1])))


def count_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  return len(inputs[0])


def make_same_attr_handler(attribute):
  def same_attr_handler(scene_struct, inputs, side_inputs):
    cache_key = '_same_%s' % attribute
    if cache_key not in scene_struct:
      cache = {}
      for i, obj1 in enumerate(scene_struct['objects']):
        same = []
        for j, obj2 in enumerate(scene_struct['objects']):
          if i != j and obj1[attribute] == obj2[attribute]:
            same.append(j)
        cache[i] = same
      scene_struct[cache_key] = cache

    cache = scene_struct[cache_key]
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    return cache[inputs[0]]
  return same_attr_handler


def make_query_handler(attribute):
  def query_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    obj = scene_struct['objects'][idx]
    assert attribute in obj
    val = obj[attribute]
    if type(val) == list and len(val) != 1:
      return '__INVALID__'
    elif type(val) == list and len(val) == 1:
      return val[0]
    else:
      return val
  return query_handler


def exist_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  assert len(side_inputs) == 0
  return len(inputs[0]) > 0


def equal_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return inputs[0] == inputs[1]


def less_than_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return inputs[0] < inputs[1]


def greater_than_handler(scene_struct, inputs, side_inputs):
  assert len(inputs) == 2
  assert len(side_inputs) == 0
  return inputs[0] > inputs[1]


def filter_ordinal_function(scene_struct, obj_idxs, o_num, o_dir):
  o_num_in_meta = ["the first one of the","the second one of the","the third one of the","the fourth one of the",
                  "the fifth one of the","the sixth one of the","the seventh one of the","the eighth one of the",
                  "the nineth one of the"]
  o_dir_in_meta = ["from left", "from right", "from front"]
  assert o_num in o_num_in_meta 
  assert o_dir in o_dir_in_meta 
  o_num = o_num_in_meta.index(o_num)
  #count the ordinal number referred object in scene_strct['object']
  objs = scene_struct['objects']
  if len(obj_idxs) == 1:
    return '__INVALID__'
  objs = [(it, objs[it]) for it in obj_idxs]
  objs_coords = [(it, obj['pixel_coords']) for it, obj in objs]

  left_to_right = sorted(objs_coords, key = lambda x: x[1][0])
  front_to_end = sorted(objs_coords, key = lambda x: x[1][2])

  if o_num >= len(objs):
    return '__INVALID__'

  if o_dir == 'from left':
    return [left_to_right[o_num][0]] #left_to_right[o_num] = [index of obj, coord]

  if o_dir == 'from right':
    return [left_to_right[-(o_num+1)][0]]

  if o_dir == 'from front':
    return [front_to_end[o_num][0]]

  assert 1==0, 'No return value'


def filter_ordinal(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  obj_idxs = inputs[0]
  o_num = side_inputs[0]
  o_dir = side_inputs[1]
  return filter_ordinal_function(scene_struct, obj_idxs, o_num, o_dir)


def get_visible_occlusion_sets(scene_struct, thre1, thre2):
  if '_obj_mask_numpy' not in scene_struct:
    scene_struct['_obj_mask_numpy'] = {}

  st1 = set([]); st2 = set([])

  objs = [(i, obj) for i, obj in enumerate(scene_struct['objects'])]
  obj_bbox = [(idx, 
               scene_struct['obj_bbox'][str(idx+1)],
               scene_struct['objects'][idx]['pixel_coords'][2]) for idx,_ in objs]

  obj_mask = [(idx, 
               scene_struct['obj_mask'][str(idx+1)],
               scene_struct['objects'][idx]['pixel_coords'][2]) for idx,_ in objs]

  def get_intersect_mask2box(scene_struct, i, j, imask, jmask, ibox, jbox, iz, jz):
    def _get_intersect_mask2box(i, j, imask, jbox, iz, jz):#let iz in front, jz in the back
      def from_imgdensestr_to_imgarray(imgstr):
        if i in scene_struct['_obj_mask_numpy']:
          return scene_struct['_obj_mask_numpy'][i]

        img = []
        cur = 0 
        num_sum=0
        for num in imgstr.split(','):
          img += [cur]*int(num);
          num_sum += int(num)
          cur = 1-cur
        if num_sum == 320*480:
          img = np.asarray(img).reshape((320,480))
        else:
          assert num_sum == 240*320
          img = np.asarray(img).reshape((240,320))
        scene_struct['_obj_mask_numpy'][i] = img
        return img

      def get_lurd(box):
        x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        l, u, r, d = x1, y1, x2, y2
        s_area = (r-l) * (d-u)
        return s_area, l, u, r, d

      imask = from_imgdensestr_to_imgarray(imask)
      sj, lj, uj, rj, dj = get_lurd(jbox)
      
      percent = np.sum(imask[uj:dj, lj:rj]) * 1.0 / sj
      return percent
    if iz < jz:
      return _get_intersect_mask2box(i,j,imask, jbox, iz, jz)
    else:
      return _get_intersect_mask2box(j,i,jmask, ibox, jz, iz)
  
  max_intersect = 0.0
  for i in range(len(obj_bbox)):
    idx, ibox, iz = obj_bbox[i]
    idx, imask, iz = obj_mask[i]
    for j in range(i+1, len(obj_bbox)):
      jdx, jbox, jz = obj_bbox[j]
      jdx, jmask, jz = obj_mask[j]
      intersect_percent = get_intersect_mask2box(scene_struct, i, j, imask, jmask, ibox, jbox, iz, jz)
      if intersect_percent > thre1 and intersect_percent < thre2:
        return '__INVALID__'
      max_intersect = max(max_intersect, intersect_percent)
      assert iz != jz
      if intersect_percent >= thre2:
        if iz > jz:
          st1.add(idx)
        elif iz < jz:
          st1.add(jdx)

  for idx,_,__ in obj_bbox:      
    if idx not in st1:
      st2.add(idx)

  if len(st1) == 0 or len(st2) == 0:
    return '__INVALID__'

  return st1, st2, max_intersect 


def filter_visibleout_function(scene_struct, obj_idxs, visible_param):
  if len(obj_idxs) == 0:
    return "__INVALID__"

  thre1 = 0.00
  thre2 = 0.20
  if '_visible_res' not in scene_struct:
    result = get_visible_occlusion_sets(scene_struct, thre1, thre2)
    if type(result) == str and result== '__INVALID__':
      return '__INVALID__'
    st1, st2, _ = result 
    scene_struct['_visible_res'] = (st1, st2)

  st1, st2 = scene_struct['_visible_res']
  st1 = (st1 & set(obj_idxs))
  st2 = (st2 & set(obj_idxs))
  assert len(st1) + len(st2) == len(obj_idxs)

  if visible_param == 'fully visible':
    return list(st2)
  else:
    return list(st1)


def filter_visibleout(scene_struct, inputs, side_inputs):
  assert len(inputs) == 1
  assert len(side_inputs) == 1
  obj_idxs = inputs[0]
  visible_param = side_inputs[0]
  assert visible_param in ["fully visible", "partially visible"]

  res = filter_visibleout_function(scene_struct, obj_idxs, visible_param)

  return res


# Register all of the answering handlers here.
# TODO maybe this would be cleaner with a function decorator that takes
# care of registration? Not sure. Also what if we want to reuse the same engine
# for different sets of node types?
execute_handlers = {
  'filter_visible': filter_visibleout,
  'filter_ordinal': filter_ordinal,
  'scene': scene_handler,
  'filter_color': make_filter_handler('color'),
  'filter_shape': make_filter_handler('shape'),
  'filter_material': make_filter_handler('material'),
  'filter_size': make_filter_handler('size'),
  'filter_objectcategory': make_filter_handler('objectcategory'),
  'unique': unique_handler,
  'relate': relate_handler,
  'union': union_handler,
  'intersect': intersect_handler,
  'count': count_handler,
  'query_color': make_query_handler('color'),
  'query_shape': make_query_handler('shape'),
  'query_material': make_query_handler('material'),
  'query_size': make_query_handler('size'),
  'exist': exist_handler,
  'equal_color': equal_handler,
  'equal_shape': equal_handler,
  'equal_integer': equal_handler,
  'equal_material': equal_handler,
  'equal_size': equal_handler,
  'equal_object': equal_handler,
  'less_than': less_than_handler,
  'greater_than': greater_than_handler,
  'same_color': make_same_attr_handler('color'),
  'same_shape': make_same_attr_handler('shape'),
  'same_size': make_same_attr_handler('size'),
  'same_material': make_same_attr_handler('material'),
}


def answer_refexp(refexp, metadata, scene_struct, all_outputs=False,
                    cache_outputs=True):
  """
  Use structured scene information to answer a structured referring expression. Most of the
  heavy lifting is done by the execute handlers defined above.

  We cache node outputs in the node itself; this gives a nontrivial speedup
  when we want to answer many referring expressions that share nodes on the same scene
  (such as during referring_expression-generation DFS). This will NOT work if the same
  nodes are executed on different scenes.
  """
  all_input_types, all_output_types = [], []
  node_outputs = []
  for node in refexp['nodes']:
    if cache_outputs and '_output' in node:
      node_output = node['_output']
    else:
      node_type = node['type']
      msg = 'Could not find handler for "%s"' % node_type
      #assert node_type in execute_handlers, msg
      handler = execute_handlers[node_type]
      node_inputs = [node_outputs[idx] for idx in node['inputs']]
      side_inputs = node.get('side_inputs', [])
      node_output = handler(scene_struct, node_inputs, side_inputs)
      if cache_outputs:
        node['_output'] = node_output
    node_outputs.append(node_output)
    if node_output == '__INVALID__':
      break

  if all_outputs:
    return node_outputs
  else:
    return node_outputs[-1]


def insert_scene_node(nodes, idx):
  # First make a shallow-ish copy of the input
  new_nodes = []
  for node in nodes:
    new_node = {
      'type': node['type'],
      'inputs': node['inputs'],
    }
    if 'side_inputs' in node:
      new_node['side_inputs'] = node['side_inputs']
    new_nodes.append(new_node)

  # Replace the specified index with a scene node
  new_nodes[idx] = {'type': 'scene', 'inputs': []}

  # Search backwards from the last node to see which nodes are actually used
  output_used = [False] * len(new_nodes)
  idxs_to_check = [len(new_nodes) - 1]
  while idxs_to_check:
    cur_idx = idxs_to_check.pop()
    output_used[cur_idx] = True
    idxs_to_check.extend(new_nodes[cur_idx]['inputs'])

  # Iterate through nodes, keeping only those whose output is used;
  # at the same time build up a mapping from old idxs to new idxs
  old_idx_to_new_idx = {}
  new_nodes_trimmed = []
  for old_idx, node in enumerate(new_nodes):
    if output_used[old_idx]:
      new_idx = len(new_nodes_trimmed)
      new_nodes_trimmed.append(node)
      old_idx_to_new_idx[old_idx] = new_idx

  # Finally go through the list of trimmed nodes and change the inputs
  for node in new_nodes_trimmed:
    new_inputs = []
    for old_idx in node['inputs']:
      new_inputs.append(old_idx_to_new_idx[old_idx])
    node['inputs'] = new_inputs

  return new_nodes_trimmed


def is_degenerate(refexp, metadata, scene_struct, answer=None, verbose=False):
  """
  A referring expression is degenerate if replacing any of its relate nodes with a scene
  node results in a referring expression with the same answer.
  """
  if answer is None:
    answer = answer_refexp(refexp, metadata, scene_struct)

  for idx, node in enumerate(refexp['nodes']):
    if node['type'] == 'relate':
      new_refexp = {
        'nodes': insert_scene_node(refexp['nodes'], idx)
      }
      new_answer = answer_refexp(new_refexp, metadata, scene_struct)
      if verbose:
        print('here is truncated referring expression:')
        for i, n in enumerate(new_refexp['nodes']):
          name = n['type']
          if 'side_inputs' in n:
            name = '%s[%s]' % (name, n['side_inputs'][0])
          print(i, name, n['_output'])
        print('new answer is: ', new_answer)

      if new_answer == answer:
        return True

  return False
