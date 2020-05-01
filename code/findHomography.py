import numpy as np
import cv2
import random

def findHomography(pts1, pts2, ransac=None,reprojThresh=4.0):
    '''
    input pts1, pts2: N x 2, N is number of 2D points
    pts1 is the target plane, means # pts1=H* pts2
    '''
    pts1 = np.hstack((pts1,np.ones((len(pts1),1))))
    pts2 = np.hstack((pts2,np.ones((len(pts2),1))))  

    # normalize
    T1,q1 = normalized2d(pts1)
    T2,q2 = normalized2d(pts2)

    if ransac==None:
        H = linearAlg(q1,q2)
        status = np.ones((len(q1),1))

    else:
        H, status = myransac(q1,q2,reprojThresh)  

    # denormalize
    H = np.linalg.inv(T1) @ H @ T2 
    H = H/H[2,2]  
    return H, status

   
def myransac(q1, q2, reprojThresh):
    '''
    input q1,q2: N x 3, N is number of 2D points
    '''
    N=len(q1)
    max_num = 0
    status = []
    for _ in range(3*N):
        inlier = []
        num=0
        index = random.sample(range(0,N),4)
        fourq1=q1[index]
        fourq2=q2[index]

        H = linearAlg(fourq1,fourq2)
        repro = (H @ q2.T).T
        for i in range(N):
            repro[i,:]=repro[i,:]/repro[i,2]

        repro_error = np.linalg.norm(q1[:,:2]-repro[:,:2], axis=1)
        for e in repro_error:
            if e < reprojThresh:
                inlier.append(1)
                num+=1
            else:
                inlier.append(0)
        # choose H with most inliers        
        if num>max_num:
            max_num=num
            status = inlier
        
    # using largest set of inliers re-compute H
    mask = [s==1 for s in status]
    sq1=q1[mask]
    sq2=q2[mask]
    H = linearAlg(sq1,sq2)

    return H, status


def linearAlg(pts1,pts2):
    '''
    input pts1,pts2: N x 3, N is number of 2D points
    '''
    B=[]
    for (q1,q2) in zip(pts1,pts2):
        # q1_mat = cv2.Rodrigues(q1)[0] #this is not right, why?
        q1_mat=np.vstack(([0,-1,q1[1]],[1,0,-q1[0]],[-q1[1],q1[0],0]))
        b = np.kron(q2, q1_mat)
        B.append(b)

    B=np.array(B).reshape((-1,9))
    u, s, vh = np.linalg.svd(B)
    v=vh.T
    hvec=v[:,-1]
    H=hvec.reshape(3,3)

    return H


def normalized2d(q):
    '''
    input q: N x 3, N is number of 2D points
    output q_norm: N x 3
    '''  
    varp = np.var(q[:,:2],axis=0)
    s=np.sqrt(1/varp)

    (deltax,deltay)= - np.mean(s*q[:,:2],axis=0)
 
    T = np.vstack(([s[0],0,deltax],[0,s[1],deltay],[0,0,1]))
    q_norm=(T @ q.T).T

    return T, q_norm


def projectpoints(K,R,t,Q):
    '''
    inout Q: N x 4, N is number of 3D points
    output q: N x 3
    '''
    Q=np.array(Q).T
    Rt=np.append(R,t,axis=1)
    project_matrix = np.dot(K,Rt) # 3x4

    q=np.dot(project_matrix,Q)
    q= (q/q[2]).T
    return q


if __name__ == "__main__":

    K = np.eye(3)
    R = np.eye(3)
    t1 = np.array([0, 0 ,0]).reshape(-1,1)
    t2 = np.array([-5, 0, 2]).reshape(-1,1)

    q1=[]
    q2=[]
    #-------------- test function projectpoints(K,R,t,Q)----------#
    for i in range(0,300,10):
        for j in range(0,400,15):
            Q=np.array([i,j,10,1])
            q1.append(projectpoints(K,R,t1,Q))
            q2.append(projectpoints(K,R,t2,Q))  

    q1_homo = np.array(q1).reshape((-1,3))
    q2_homo = np.array(q2).reshape((-1,3))

    q1=q1_homo[:,:2]
    q2=q2_homo[:,:2]

    #---------------test function normalized2d(p)------------------#

    # T,q1 = normalized2d(q1)
    # print(q1)
    # print(np.mean(q1,axis=0))
    # print(np.var(q1,axis=0))

    # #--------------test function linearAlg(pts1,pts2)---------#
    # H = findHomography(q1,q2)
    # q1_re = (H @ q2.T).T
    # for i in range(len(q1_re)):
    #     q1_re[i,:]=q1_re[i,:]/q1_re[i,2]
    # print(H)
    # print(q1)
    # print(q1_re)

    #-------------test function ransac(pts1, pts2) ----------------#

    H, status = findHomography(q1,q2,ransac=1,reprojThresh=2)
    #H, status= findHomography(q1,q2)
    q1_homo_re= (H @ q2_homo.T).T
    for i in range(len(q1_homo_re)):
        q1_homo_re[i,:]=q1_homo_re[i,:]/q1_homo_re[i,2]

    repro_error = np.sum(np.linalg.norm(q1-q1_homo_re[:,:2], axis=1)) 
    print(H)
    #print(status)
    print(repro_error)


    #------------- the opencv function -------------------------#

    (H, status) = cv2.findHomography(q2, q1, cv2.RANSAC, 2)
    #(H, status) = cv2.findHomography(q2, q1)
    q1_homo_re= (H @ q2_homo.T).T
    for i in range(len(q1_homo_re)):
        q1_homo_re[i,:]=q1_homo_re[i,:]/q1_homo_re[i,2]

    repro_error = np.sum(np.linalg.norm(q1-q1_homo_re[:,:2], axis=1)) 
    print(H)
    #print(status)
    print(repro_error)
