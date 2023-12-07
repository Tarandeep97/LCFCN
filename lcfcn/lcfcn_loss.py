import torch
import skimage
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
from skimage import morphology as morph
import cv2
from scipy.spatial.distance import euclidean



def compute_loss(points, probs, roi_mask=None):
    """
    images: n x c x h x w
    probs: h x w (0 or 1)
    """
    points = points.squeeze()
    probs = probs.squeeze()

    assert(points.max() <= 1)

    tgt_list = get_tgt_list(points, probs, roi_mask=roi_mask)
    
    # image level
    # pt_flat = points.view(-1)
    pr_flat = probs.view(-1)
    
    # compute loss
    loss = 0.
    for tgt_dict in tgt_list:
        pr_subset = pr_flat[tgt_dict['ind_list']]
        # pr_subset = pr_subset.cpu()
        loss += tgt_dict['scale'] * F.binary_cross_entropy(pr_subset, 
                                        torch.ones(pr_subset.shape, device=pr_subset.device) * tgt_dict['label'], 
                                        reduction='mean')
        
    return loss

@torch.no_grad()
def get_tgt_list(points, probs, roi_mask=None):
    tgt_list = []

    # image level
    pt_flat = points.view(-1)
    pr_flat = probs.view(-1)

    u_list = points.unique()
    #print(u_list)
    if 0 in u_list:
        ind_bg = pr_flat.argmin()
        tgt_list += [{'scale': 1, 'ind_list':[ind_bg], 'label':0}]   

    if 1 in u_list:
        ind_fg = pr_flat.argmax()
        tgt_list += [{'scale': 1, 'ind_list':[ind_fg], 'label':1}]   

    # point level
    if 1 in u_list:
        ind_fg = torch.where(pt_flat==1)[0]
        tgt_list += [{'scale': len(ind_fg), 'ind_list':ind_fg, 'label':1}]  

    #print(tgt_list)
    # get blobs
    probs_numpy = probs.detach().cpu().numpy()
    blobs = get_blobs(probs_numpy, roi_mask=None)
    # get foreground and background blobs
    points = points.cpu().numpy()
    fg_uniques = np.unique(blobs * points)
    bg_uniques = [x for x in np.unique(blobs) if x not in fg_uniques]

    # split level
    # -----------
    n_total = points.sum()

    if n_total > 1:
        # global split
        boundaries = watersplit(probs_numpy, points)
        ind_bg = np.where(boundaries.ravel())[0]

        tgt_list += [{'scale': (n_total-1), 'ind_list':ind_bg, 'label':0}]  

        # local split
        for u in fg_uniques:
            if u == 0:
                continue

            ind = blobs==u

            b_points = points * ind
            n_points = b_points.sum()
            
            if n_points < 2:
                continue
            
            # local split
            boundaries = watersplit(probs_numpy, b_points)*ind
            ind_bg = np.where(boundaries.ravel())[0]

            tgt_list += [{'scale': (n_points - 1), 'ind_list':ind_bg, 'label':0}]  

    # fp level
    for u in bg_uniques:
        if u == 0:
            continue
        
        b_mask = blobs==u
        if roi_mask is not None:
            b_mask = (roi_mask * b_mask)
        if b_mask.sum() == 0:
            pass
            # from haven import haven_utils as hu
            # hu.save_image('tmp.png', np.hstack([blobs==u, roi_mask]))
            # print()
        else:
            ind_bg = np.where(b_mask.ravel())[0]
            tgt_list += [{'scale': 1, 'ind_list':ind_bg, 'label':0}]  

    return tgt_list 



def watersplit(_probs, _points, flag=1):
    points = _points.copy()

    points[points != 0] = np.arange(1, points.sum()+1)
    points = points.astype(float)

    probs = ndimage.black_tophat(_probs.copy(), 7)
    seg = watershed(probs, points)

    if (flag):
        return find_boundaries(seg)
    else:
        return seg


def get_blobs(probs, roi_mask=None):
    probs = probs.squeeze()
    h, w = probs.shape
 
    pred_mask = (probs>0.5).astype('uint8')
    blobs = np.zeros((h, w), int)

    blobs = morph.label(pred_mask == 1)

    if roi_mask is not None:
        blobs = (blobs * roi_mask[None]).astype(int)

    return blobs


def blobs2points(blobs):
    blobs = blobs.squeeze()
    points = np.zeros(blobs.shape).astype("uint8")
    rps = skimage.measure.regionprops(blobs.astype(int))

    assert points.ndim == 2

    for r in rps:
        y, x = r.centroid
        points[int(y), int(x)] = 1

    return points

def compute_game(pred_points, gt_points, L=1):
    n_rows = 2**L
    n_cols = 2**L

    pred_points = pred_points.astype(float).squeeze()
    gt_points = np.array(gt_points).astype(float).squeeze()
    h, w = pred_points.shape
    se = 0.

    hs, ws = h//n_rows, w//n_cols
    for i in range(n_rows):
        for j in range(n_cols):

            sr, er = hs*i, hs*(i+1)
            sc, ec = ws*j, ws*(j+1)

            pred_count = pred_points[sr:er, sc:ec]
            gt_count = gt_points[sr:er, sc:ec]
            
            se += float(abs(gt_count.sum() - pred_count.sum()))
    return se

def save_tmp(fname, images, logits, radius, points):
    from haven import haven_utils as hu
    probs = F.softmax(logits, 1); 
    mask = probs.argmax(dim=1).cpu().numpy().astype('uint8').squeeze()
    img_mask=hu.save_image('tmp2.png', 
                hu.denormalize(images, mode='rgb'), 
                mask=mask, return_image=True)
    hu.save_image(fname,np.array(img_mask)/255. , radius=radius,
                    points=points)

def get_random_points(mask, n_points=1, seed=1):
    from haven import haven_utils as hu
    y_list, x_list = np.where(mask)
    points = np.zeros(mask.squeeze().shape)
    with hu.random_seed(seed):
        for i in range(n_points):
            yi = np.random.choice(y_list)
            x_tmp = x_list[y_list == yi]
            xi = np.random.choice(x_tmp)
            points[yi, xi] = 1

    return points

def get_points_from_mask(mask, bg_points=0):
    n_points = 0
    points = np.zeros(mask.shape)
    # print(np.unique(mask))
    assert(len(np.setdiff1d(np.unique(mask),[0,1,2] ))==0)

    for c in np.unique(mask):
        if c == 0:
            continue
        blobs =  morph.label((mask==c).squeeze())
        points_class = blobs2points(blobs)

        ind = points_class!=0
        n_points += int(points_class[ind].sum())
        points[ind] = c
    assert morph.label((mask).squeeze()).max() == n_points
    points[points==0] = 255
    if bg_points == -1:
       bg_points = n_points

    if bg_points:
        from haven import haven_utils as hu
        y_list, x_list = np.where(mask==0)
        with hu.random_seed(1):
            for i in range(bg_points):
                yi = np.random.choice(y_list)
                x_tmp = x_list[y_list == yi]
                xi = np.random.choice(x_tmp)
                points[yi, xi] = 0

    return points
        

def find_max_pixels(probs):
    #Create contours
    probs = (probs>0.5).astype(np.uint8)
    blob_contours, _ = cv2.findContours(probs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_count = len(blob_contours)
    max_pixel_prob = np.zeros(probs.shape)
    
    for i in range(contours_count):
        #Select a prob blob
        contour_image = np.zeros(probs.shape)
        cand_contour = cv2.drawContours(contour_image, [blob_contours[i]], 0, 1, thickness=cv2.FILLED)
        cand_contour[cand_contour==255]=1
        probs_contour = cand_contour*probs
        
        #Normalize Probs
        probs_contour = probs_contour/np.sum(probs_contour)
        
        max_index = np.unravel_index(np.argmax(probs_contour), probs_contour.shape)
        
        x,y = max_index
        max_pixel_prob[y,x]=1
        
        
    return max_pixel_prob


def find_centroid_pixels(probs):
    #Create contours
    probs = (probs>0.5).astype(np.uint8)
    blob_contours, _ = cv2.findContours(probs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_count = len(blob_contours)
    centroid_pixels = np.zeros(probs.shape)
    
    for i in range(contours_count):
        #Select a prob blob
        contour_image = np.zeros(probs.shape)
        cand_contour = cv2.drawContours(contour_image, [blob_contours[i]], 0, 1, thickness=cv2.FILLED)
        cand_contour[cand_contour==255]=1
        
        M = cv2.moments(cand_contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        
        
        centroid_pixels[cy,cx]=1
        
        
    return centroid_pixels

def non_zero_pixel_positions(image):
    non_zero_pixels = np.array(np.where(image > 0)).T
    return non_zero_pixels

def compute_distances(point, pixel_positions):
    distances = []
    for j in range(0, len(pixel_positions)):
        distance = euclidean(point, pixel_positions[j])
        distances.append(distance)
        
    return distances


def psll2_max(points, mask, probs, mode="max"):
    
    """
    mode : default as "max"  or "centroid"
    """
    #Generate Watershed splits using points and mask
  
    points = points.squeeze()
    mask = mask.squeeze()
    probs = probs.squeeze()
    
    watershed_split = watersplit(mask,points, flag=0)
    
    #Select each split
    split_count = len(set(watershed_split.flatten()))
    
    if (mode=="max"):
        selected_pixels = find_max_pixels(probs)
    elif (mode=="centroid"):
        selected_pixels = find_centroid_pixels(probs)
        
    
    #merge_gtpred_points = np.logical_or(points,selected_pixels) 
    
    pairwise_dist = 0
    
    for idx in range(split_count):
        split = (watershed_split==idx+1).astype(np.uint8)
        
        #select points within the split
        max_prob_points_pos = non_zero_pixel_positions(np.logical_and(split, selected_pixels))
        point_pos = non_zero_pixel_positions(np.logical_and(split,points))
        
        if (len(point_pos)==0):
            continue
            
        if (len(max_prob_points_pos)==0):
            split_boundary = find_boundaries(split)
            split_boundary_pos = non_zero_pixel_positions(split_boundary)
            if (len(split_boundary_pos)==0):
                continue
            
            np_dist = euclidean(split_boundary_pos[0], point_pos)
            
        else:
            np_dist = np.array(compute_distances(point_pos, max_prob_points_pos))
            
        pairwise_dist+= np.min(np_dist)
        
    avg_pairwise_dist = pairwise_dist/split_count
    
    return avg_pairwise_dist

    

    
    
                
    
                
    
    