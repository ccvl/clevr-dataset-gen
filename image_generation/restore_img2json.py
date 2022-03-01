import cv2
import sys
import os
import json

def main(index, idx):
  obj_mask = {}
  for _i in range(index):
    path = './tmp_graph_output/indexob{}_{}/Image0001.png'.format(_i,idx)
    print('read path...', path)
    img = cv2.imread(path)[:,:,0].flatten().tolist()
    img = [int(_x==255) for _x in img]
    def to_str(img):
      pre = 0
      has = 0
      lis = []
      for _x in img:
        if _x == pre:
          has += 1
        else:    
          lis.append(has)
          pre = 1-pre
          has = 1
      if has != 0:
        lis.append(has)
      return ','.join([str(it) for it in lis])
    str_img = to_str(img)
    obj_mask[_i+1] = str_img
    path = './tmp_graph_output/indexob{}_{}/'.format(_i, idx)
    os.system('rm -r {}'.format(path))

  json.dump(obj_mask, open('/tmp/obj_mask_{}.json'.format(idx), 'w'))


main(int(sys.argv[1]), int(sys.argv[2]))
