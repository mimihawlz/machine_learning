# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self,x):
        self.val = x
        self.next = None
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    ##返回合并后列表
    """输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。"""
    def Merge(self,pHead1,pHead2):
        res = []
        while pHead1:
            res.append(pHead1.val)
            pHead1 = pHead1.next
        while pHead2:
            res.append(pHead2.val)
            pHead2 = pHead2.next
        res.sort()
        ##新建一个list 将链表的val进行排序
        ## 重新建立一个arraylist 将
        #init head == dummy == null
        dummy = ListNode(0)
        pre = dummy
        for i in res:
            node = ListNode(i)
            pre.next = node
            pre = pre.next
        return dummy.next
        #dummy == null


    """树的子结构"""
    """输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）"""

    def HasSubTree(self,pRoot1,pRoot2):

        def convert(p):
            if p:
                return str(p.val) + convert(p.left) + convert(p.right)
            else:
                return " "
        return convert(pRoot2) in convert(pRoot1) if pRoot2 else False

    '''
       思路：先判断b是否是a的子结构 首先第一步在树A中查找与B根结点值一样的节点
       第二步：判断树a中以R为根结点的子树是否和树B具有相同的结构
       值相同 递归
       终止条件：达到树A或者树B的叶节点
    '''
    def HasSubTree1(self,pRoot1,pRoot2):
        result = False
        if pRoot1  is not None and pRoot2 is not None:
            if pRoot2.val == pRoot1.val:
                result = self.doesTree1haveTree2(pRoot1,pRoot2)
            ##递归求解下一个子节点
            if not result:
                result = self.HasSubTree(pRoot1.left,pRoot2)
            if not result:
                result = self.HasSubTree(pRoot1.right,pRoot2)
        return  result

    #判断节点是否相同
    #注意前两个if不能换顺序
    # 若颠倒顺序 是先判断pRoot1 意味着pRoot2已经完成遍历了 但是返回的False 判断出错
    def doesTree1haveTree2(self,pRoot1,pRoot2):
        if pRoot2 is None:
            return  True
        if pRoot1 is None:#root1 没有匹配root2的值
            return False
        if pRoot1.val != pRoot2.val:
            return  False
        '''递归求值'''
        return self.doesTree1haveTree2(pRoot1.left,pRoot2.left) and self.doesTree1haveTree2(pRoot1.right,pRoot2.right)

    '''操作给定的二叉树，将其变换为源二叉树的镜像。'''
    ''' 源二叉树 
     8
    / \
   6    10
  / \   / \
5    7 9  11
镜像二叉树
        8
       / \
     10    6
    /  \  / \
   11  9 7   5
'''
    '''
    先交换两个根结点的值 接着递归求解即可
    '''
    def Mirror(self,root):
        if not root:
            return  root
        node = root.left
        root.left = root.right
        root.right = node
        ###
        #两变量交换可以这样写    root.left,root.right = root.right,root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root


    '''
    顺时针打印矩阵
    
    输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
    例如，如果输入如下4 X 4矩阵： 
    1 2 3 4 
    5 6 7 8 
    9 10 11 12 
    13 14 15 16 
    则依次打印出数字
    1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
    '''
    '''
    可以模拟魔方逆时针旋转的方法，一直做取出第一行的操作
    例如 
    1 2 3
    4 5 6
    7 8 9
    输出并删除第一行后，再进行一次逆时针旋转，就变成：
    6 9
    5 8
    4 7
    继续重复上述操作即可。
    '''

    def printMatrix(self, matrix):
        result = []
        while(matrix):
            result += matrix.pop(0)#把该行所有数字加入进去 并且去除该行
            if not matrix or not matrix[0]:
                break
            matrix = self.turn(matrix)#翻转
        return  result

    def turn(self,matrix):
        num_r = len(matrix)
        num_c = len(matrix[0])
        newmat = []
        for i in range(num_c):
            newmat2 = []
            for j in range(num_r):
                newmat2.append(matrix[j][i])#read data in column
            newmat.append(newmat2)#read data in row
        newmat.reverse()#reserver 按照中心对称
        return newmat

    """
    包含min函数的栈
    
    定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数
   （时间复杂度应为O（1））。
    """

    """
    思路：利用一个辅助栈来存放最小值

    栈  3，4，2，5，1
    辅助栈 3，3，2，2，1
    
    每入栈一次，就与辅助栈顶比较大小，如果小就入栈，如果大就入栈当前的辅助栈顶
    当出栈时，辅助栈也要出栈
    这种做法可以保证辅助栈顶一定都当前栈的最小值
    """

    def __init__(self):
        self.stack = []
        self.assist = []


    def push(self,node):
        min = self.min()
        #假设assist还没有初始化 or 最小值< node
        if not min or min > node:
            self.assist.append(node)
        else:
            self.assist.append(min)
        self.stack.append(node)

    '''
    将assist和stack都要pop 出栈
    '''
    def pop(self):
        if self.stack:
            self.assist.pop()
            return  self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1]


    def min(self):
        if self.assist:
            return  self.assist[-1]


    '''
    栈的压入和弹出序列
    
    输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
    假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，
    序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
    （注意：这两个序列的长度是相等的）
    
    入栈：12345
    出栈：45321
    '''

    '''
    思路：借用一个辅助栈 遍历压栈顺序，第一个入栈顺序是1， 判断栈顶元素是否出栈顺序的第一个元素 1！=4
    继续压栈 知道相等后开始出栈 4 == 4 将出栈元素向后移动一位 知道不相等
    
    
入栈1,2,3,4,5

出栈4,5,3,2,1

首先1入辅助栈，此时栈顶1≠4，继续入栈2

此时栈顶2≠4，继续入栈3

此时栈顶3≠4，继续入栈4

此时栈顶4＝4，出栈4，弹出序列向后一位，此时为5，,辅助栈里面是1,2,3

此时栈顶3≠5，继续入栈5

此时栈顶5=5，出栈5,弹出序列向后一位，此时为3，,辅助栈里面是1,2,3

….

依次执行，最后辅助栈为空。如果不为空说明弹出序列不是该栈的弹出顺序
    '''

    def IsPopOrder(self, pushV, popV):
        if len(pushV) == 0  or len(popV) == 0:
            return False
        #模拟stack中存入pushV的数据
        stack = []
        #用于识别弹出序列的位置
        sindex = 0
        for i in range(0,len(pushV)):
            stack.append(pushV[i])
            while sindex<len(pushV) and stack[-1] == popV[sindex] and  stack :
                ###注意 没有添加sindex<len(pushV) 导致list out of range
              stack.pop()
              sindex += 1
        if stack :
            return  False
        return  True


if __name__ == '__main__':
    a = Solution()
    pushV = [1,2,3,4,5]
    popV = [4,5,3,2,1]
    b = a.IsPopOrder(pushV,popV)
    print(b)


