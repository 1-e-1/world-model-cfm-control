"""Primitive geometry shared by sampling, simulation and offline GT metrics.

Never imported by ActiveTSDF or its next-view selector. Sizes are local full
extents in metres: box (x,y,z), sphere (d,d,d), cylinder (d,d,height).
"""
from __future__ import annotations

import numpy as np

SHAPE_NAMES = ("cuboid", "sphere", "cylinder")


def sample_shapes_and_sizes(count, rng, minimum=(.06,.06,.08), maximum=(.16,.16,.24),
                           shape_names=SHAPE_NAMES):
    lower,upper=np.asarray(minimum,dtype=float),np.asarray(maximum,dtype=float)
    if count < 1 or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("expected positive count and three size bounds")
    if not np.isfinite([lower,upper]).all() or not ((upper>lower)&(lower>0)).all():
        raise ValueError("invalid size bounds")
    if not shape_names or len(set(shape_names)) != len(shape_names) or any(s not in SHAPE_NAMES for s in shape_names):
        raise ValueError("unknown or duplicate shape family")
    allowed=np.asarray([SHAPE_NAMES.index(s) for s in shape_names],dtype=np.int32)
    # Do not force every family into every scene: each object is an iid draw.
    ids=rng.choice(allowed,size=count)
    sizes=rng.uniform(lower,upper,size=(count,3))
    for i,shape in enumerate(ids):
        axes=3 if shape==1 else 2
        if shape:
            lo,hi=lower[:axes].max(),upper[:axes].min()
            if hi <= lo:
                raise ValueError("size ranges cannot support a round primitive")
            sizes[i,:axes]=rng.uniform(lo,hi)
    return sizes.astype(np.float32),ids


def shape_half_extent(size, rotation, shape_id=0):
    size=np.asarray(size)
    if shape_id==0:
        return np.abs(rotation)@(size*.5)
    if shape_id==1:
        return np.full(3,size[0]*.5)
    if shape_id==2:
        return np.linalg.norm(rotation[:,:2],axis=1)*size[0]*.5+np.abs(rotation[:,2])*size[2]*.5
    raise ValueError(f"unknown shape id {shape_id}")


def sample_local_surface(size, shape_id, count, rng):
    """Area-uniform primitive surface samples, not uniformly chosen box faces."""
    size=np.asarray(size,dtype=float)
    if count<1 or size.shape != (3,) or not (size>0).all():
        raise ValueError("invalid surface sample request")
    if shape_id==0:
        areas=np.asarray([size[1]*size[2],size[0]*size[2],size[0]*size[1]])
        axis=rng.choice(3,size=count,p=areas/areas.sum())
        points=rng.uniform(-.5,.5,size=(count,3))*size
        points[np.arange(count),axis]=rng.choice([-1.,1.],size=count)*size[axis]*.5
    elif shape_id==1:
        points=rng.normal(size=(count,3))
        points=points/np.linalg.norm(points,axis=1,keepdims=True)*size[0]*.5
    elif shape_id==2:
        radius,height=size[0]*.5,size[2]
        side=rng.random(count)<height/(height+radius)
        angle=rng.uniform(-np.pi,np.pi,size=count)
        radial=np.where(side,radius,radius*np.sqrt(rng.random(count)))
        z=np.where(side,rng.uniform(-height*.5,height*.5,size=count),
                   rng.choice([-1.,1.],size=count)*height*.5)
        points=np.stack([radial*np.cos(angle),radial*np.sin(angle),z],1)
    else:
        raise ValueError(f"unknown shape id {shape_id}")
    return points.astype(np.float32)


def signed_distance_to_primitives(points, centers, sizes, quaternions_wxyz, shape_ids):
    """Exact analytic SDF for separated boxes, spheres and finite cylinders.

    points [N,3], object arrays [M,3]/[M,4]/[M]; output [N]. GT evaluation only.
    """
    import torch
    from codex_zbot.perception.pointcloud import quaternion_to_matrix
    ids=torch.as_tensor(shape_ids,device=points.device,dtype=torch.long)
    if ids.shape != (len(centers),) or ((ids<0)|(ids>=len(SHAPE_NAMES))).any():
        raise ValueError("invalid shape IDs")
    rotation=quaternion_to_matrix(quaternions_wxyz)
    local=torch.einsum('nmi,mij->nmj',points[:,None]-centers[None],rotation)
    q=local.abs()-sizes[None]*.5
    distance=q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
    sphere=local.norm(dim=-1)-sizes[None,:,0]*.5
    radial=local[...,:2].norm(dim=-1)-sizes[None,:,0]*.5
    axial=local[...,2].abs()-sizes[None,:,2]*.5
    cylinder_q=torch.stack([radial,axial],-1)
    cylinder=cylinder_q.clamp_min(0).norm(dim=-1)+cylinder_q.amax(-1).clamp_max(0)
    distance=torch.where(ids[None]==1,sphere,torch.where(ids[None]==2,cylinder,distance))
    return distance.amin(1)
