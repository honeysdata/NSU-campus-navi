"""Build a lightweight campus graph JSON from any OSM XML extract."""
import argparse, json, math, xml.etree.ElementTree as ET
from pathlib import Path

def build(source):
    root=ET.parse(source).getroot(); nodes={int(n.attrib['id']):(float(n.attrib['lon']),float(n.attrib['lat'])) for n in root.findall('node')}
    allowed={'footway','path','pedestrian','steps','service','track','cycleway'}; edges=[]; seen=set(); used=set()
    for w in root.findall('way'):
        tags={t.attrib['k']:t.attrib['v'] for t in w.findall('tag')}
        if tags.get('highway') not in allowed: continue
        refs=[int(x.attrib['ref']) for x in w.findall('nd') if int(x.attrib['ref']) in nodes]
        for a,b in zip(refs,refs[1:]):
            key=tuple(sorted((a,b)))
            if a==b or key in seen: continue
            seen.add(key); used.update((a,b)); edges.append({'source':a,'target':b,'coords':[nodes[a],nodes[b]],'highway':tags['highway'],'weight':1.0})
    coords=[nodes[n] for n in used]; center=[sum(y for _,y in coords)/len(coords),sum(x for x,_ in coords)/len(coords)]
    return {'center':center,'zoom':15,'nodes':[{'id':n,'lon':nodes[n][0],'lat':nodes[n][1]} for n in sorted(used)],'edges':edges,'training_metrics':{'epochs':0,'seconds':None,'seconds_per_epoch':None},'training':{'epochs':[],'loss':[]},'steps':[]}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(build(a.source),ensure_ascii=False),encoding='utf-8');print(out)
