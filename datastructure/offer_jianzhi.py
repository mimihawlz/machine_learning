'''
二维数组中的查找问题
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，
判断数组中是否含有该整数。
'''

'''
解决思路:从左下角元素往上查找 右边元素>这个元素 上边元素<这个元素
target<array[j][i] 往上面找
target>array[j][i]  往右面找
如果出了边界 False
'''
class Solution:
    # array 二维列表
    def Find(selfself,target,array):
        if array == []:
            return False
        num_row = len(array)
        num_column = len(array[0])

        i = num_column - 1# column
        j = 0 # row

        while i >= 0 and j < num_column:
            if array[j][i] > target:
                i -= 1
            elif array[j][i] < target:
                j += 1
            else:
                return True
        return False
    """
    替换空格:
    请实现一个函数，将一个字符串中的每个空格替换成“%20”。
    例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
    
    解决思路:创建一个空字符串 存放相应的结果
    """


    def replaceSpace(self, s):
        new_s = ' '
        for i in s:#遍历字符串
            if  i == ' ':
                new_s = new_s + '%20'#如果为空 添加
            else:
                new_s = new_s + i#不为空 满足


    """
        3.从尾到头打印链表
    输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
    解决思路:新建一个result 若有列表，
    
    """


    #初始化listNode值
    class listNode:
        def __init__(self,x):
            self.val = x
            self.next = None
    def printListFromTailToHead(self, listNode):
        result = []
        if listNode is None:
            return result
        while listNode is not None:
            result.extend([listNode.val])
            listNode = listNode.next
        result.extend([listNode.val])#extend返回需要list

        return result[::-1]#反向链表
    #s = 'abcdefg'
    #s[::-1] 'gfedcba'
    #s[::2] 'aceg'
    #s[6:2:-2] 'ge'
    #s[:3] 'abc'

    """
        4.重建二叉树
    
    输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
    假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
    例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，
    则重建二叉树并返回。
    """


    class TreeNode:
        def __init__(self,x):
            self.val = x
            self.left = None
            self.right = None
    def reConstructBinaryTree(self, pre, tin):
        if len(pre) == 0:
            return  None
        elif len(pre) == 1:
            return TreeNode(pre[0])
        else:
            ans = TreeNode(pre[0])
            ans.left = self.reConstructBinaryTree(pre[1:tin.index(pre[0])+1],tin[:tin.index(pre[0])])
            ans.right = self.reConstructBinaryTree(pre[tin.index(pre[0])+1:],tin[tin.index(pre[0])+1:])



    """
      5.用两个栈实现队列
    用两个栈来实现一个队列，完成队列的Push和Pop操作。 
    队列中的元素为int类型。
    思路:元素依次进入栈1,再从栈1依次弹出到栈2，然后弹出栈2顶部元素 
    """

    def __init__(self):
        self.stackA = []
        self.stackB = []
    def push(self,node):
        self.stackA.append(node)
    def pop(self):
        if self.stackB:#stackB不为空 直接弹出stackB
            return self.stackB.pop()
        elif not self.stackA:# 如果stackA没有元素 返回None
            return None
        else:#stackA不为空且stackB为空
            while self.stackA:#
                self.stackB.append(self.stackA.pop())
                return self.stackB.pop()
  """
    旋转数组的最小数字
  把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
  输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 
  例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
  NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
  """

    def minNumberInRotateArray(self, rotateArray):
        if not rotateArray:
          return 0
        else:
          return min(rotateArray)

  """
    斐波那契数列:
  大家都知道斐波那契数列，现在要求输入一个整数n，
  请你输出斐波那契数列的第n项（从0开始，第0项为0）。
  n<=39
  """


    def Fibonacci(self, n):
      res = [0,1,1,2]
      while len(res) <= n:
          res.append(res[-1]+res[-2])
      return res[n]

    '''
    跳台阶:
    一只青蛙一次可以跳上1级台阶，也可以跳上2级。
    求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
    思路:建立一个res 表示每次对应的台阶个数的方法
    '''
    def jumpFloor(self, number):
        res = [1,1,2]#分别有1个方法 1个方法 2个方法
        while len(res) <= number:
            res.append(res[-1] + res[-2])
        return  res[number]

    """
    一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。
    求该青蛙跳上一个n级的台阶总共有多少种跳法。
    解析:观察每个台阶对应的方法数
    """
    def jumpFloorII(self, number):
        return 2**(number-1)

    """
    矩形覆盖:
    我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
    请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
    
    """

    def rectCover(self, number):
        res = [0,1,2]
        while len(res) <= number:
            res.append(res[-1]+res[-2])
        return res[number]

    """ 二进制中1的个数:
    输入一个整数，输出该数二进制表示中1的个数 其中负数用补码表示 """
   """
       如果n>0 直接计算即可 如果n<0 计算2**32 + n的结果中1的个数

   """

    def NumberOf1(self, n):
        if n>=0:
             return bin(n).replace("0b","").count("1")
        else:
             return bin(2**32 + n).count("1")


    """
    数值的整数次方
    给定一个double类型的浮点数base和int类型的整数exponent。
    求base的exponent次方。
    """
def Power(self, base, exponent):
    return  base**exponent

    """
    调整数组顺序使奇数位于偶数前面
    
    输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
    所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
    """

    def reOrderArray(self, array):
        odd,even = [],[]
        for i in  array:
            if i % 2 == 1:
                odd.append(i)
            else:
                even.append(i)
        return odd+even
    """
    链表中倒数第k个结点
    
    输入一个链表,输出该链表中倒数第k个结点
    """
    class ListNode:
        def __init__(self,x):
            self.val = x
            self.next = None

    def FindKthToTail(self, head, k):
        res = []
        while head:
            res.append(head)
            head = head.next
        if k> len(res) or k < 1:
            return
        return res[-k]

    """
    反转链表
    
    输入一个链表 反转链表后 输出新链表的表头
    """
    def ReverseList(self, pHead):
        if not pHead or not pHead.next:
            return pHead#始终指向要反转的结点

        last = None#指向反转后的首结点

        # 1 -> 2
        # tmp = 2 pHead->next = None last = 1 pHead = 2
        # tmp = None pHead->next = 1 last = 2 pHead = None
        while pHead:#始终指向要反转的结点
            tmp = pHead.next#
            pHead.next = last
            last = pHead
            pHead = tmp
        return  last




















