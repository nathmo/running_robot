import xml.etree.ElementTree as ET
from pathlib import Path

def normalize(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())

mt=ET.parse('mujoco/running_robot_debug.xml')
axes={}
for body in mt.findall('.//body'):
    bname=body.get('name','')
    norm=normalize(bname)
    for joint in body.findall('joint'):
        axis=joint.get('axis')
        if axis:
            axes[norm]=axis
            break
print('mujoco axes count', len(axes))
print(list(axes.items())[:20])

u=ET.parse('robotURDF/urdf/Assy_Full_Aligned_URDF.urdf')
children=[]
for j in u.findall('joint'):
    c=j.find('child')
    if c is not None:
        name=c.get('link')
        children.append((name, normalize(name)))
print('urdf child count', len(children))
print(children[:30])
