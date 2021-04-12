class Heap:
    def __init__(self):
        self.heap = [0]
        self.max = 0
        self.multiplier = 300

    def length(self):
        return len(self.heap) - 1

    # This heap function appends the new element to the array and calls sift_up to position it. Sift_up runs in Log n
    # time, so insert likewise runs in log n time and constant space complexity.
    def insert(self, element):
        self.heap.append(element)
        if len(self.heap) - 1 > self.max:
            self.max = len(self.heap) - 1
        self.sift_up(len(self.heap) - 1)

    # This heap function saves the root node (the smallest distance value) and replaces it with the last node in the
    # array. It then sifts_down that last node by calling sift_down before returning the smallest previous root.
    # Since sift_node runs in log n time and constant space, remove() does as well
    def remove(self):
        if len(self.heap) == 1:
            return None
        root = self.heap[1]
        self.heap[1] = self.heap[len(self.heap) - 1]
        self.heap.pop()
        self.sift_down(1)
        return root

    # This Heap operation moves the element at the argument position up in the binary heap to its proper position It
    # runs in O(log(n)) because WCS it will go from the very bottom to the very top of the tree, which is log(n) moves
    def sift_up(self, pos):
        # update the index map
        # Loop until the index is before the root of the tree
        while pos // 2 > 0:
            # If the distance value of the current node, swap its position with its parent, and update the index map

            if self.score(pos) < self.score(pos // 2):
                self.heap[pos], self.heap[pos // 2] = self.heap[pos // 2], self.heap[pos]
                pos = pos // 2
            else:
                break

    # This Heap operation moves the element at the argument position down in the binary heap to its proper position
    # It runs in O(log(n)) because WCS it will go from the very top to the very bottom of the tree, which is log(n)
    # moves
    def sift_down(self, pos):
        # Loop until the potential children of the index no longer exist in the array
        while (pos * 2) <= len(self.heap) - 1:
            #  Uses the min_child function to determine which of its two children is smaller.
            mc = self.min_child(pos)
            # if the distance value of the parent is larger then that of the child, swap their places in the heap,
            # and update the heap map accordingly

            if self.score(pos) > self.score(mc):
                self.heap[pos], self.heap[mc] = self.heap[mc], self.heap[pos]
                pos = mc
            else:
                break

    # This function is used exclusively by sift_down to see which of its children is the smallest. It runs in
    # constant time, only having to check at most two indexes in the heap, and comparing their distance values.
    def min_child(self, pos):
        # The constraints in sift_down assure the element has at least 1 child, if it only has 1, return it
        if (pos * 2) + 1 > len(self.heap) - 1:
            return pos * 2
        # If it has more then 1 child, return the one with the smaller distance value
        else:
            if self.score(pos * 2) < self.score((pos * 2) + 1):
                return pos * 2
            else:
                return (pos * 2) + 1

# Score is calculated by giving some level of weight (stipulated by the multiplier defined above) and prioritizing
    # the lowerBound after subtracting off the weight given to lower level states
    def score(self, pos):
        b = self.heap[pos].lowerBound
        l = self.heap[pos].level
        return b - (l * self.multiplier)


