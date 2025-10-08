#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
intervaltree: A mutable, self-balancing interval tree for Python 2 and 3.
Queries may be by point, by range overlap, or by range envelopment.

Core logic: internal tree nodes.

Copyright 2013-2018 Chaim Leib Halbert
Modifications Copyright 2014 Konstantin Tretyakov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from operator import attrgetter
from math import floor, log
import cython

@cython.cfunc
@cython.returns(cython.double)
@cython.locals(num=cython.double)
def l2(num):
    """
    log base 2
    :rtype real
    """
    return log(num, 2)

@cython.cclass
class Node(object):
    x_center = cython.declare(object)
    s_center = cython.declare(set)
    left_node = cython.declare(object, visibility='readonly')
    right_node = cython.declare(object, visibility='readonly')
    depth = cython.declare(cython.int)
    balance = cython.declare(cython.int)

    def __init__(self, x_center=None, s_center=None, left_node=None, right_node=None):
        if s_center is None:
            s_center = set()
        self.x_center = x_center
        self.s_center = set(s_center)
        self.left_node = left_node
        self.right_node = right_node
        self.depth = 0
        self.balance = 0
        self.rotate()

    @classmethod
    def from_interval(cls, interval):
        center = interval.begin
        return Node(center, [interval])

    @classmethod
    def from_intervals(cls, intervals):
        if not intervals:
            return None
        return Node.from_sorted_intervals(sorted(intervals))

    @classmethod
    def from_sorted_intervals(cls, intervals):
        if not intervals:
            return None
        node = Node()
        node = node.init_from_sorted(intervals)
        return node

    def init_from_sorted(self, intervals):
        center_iv = intervals[len(intervals) // 2]
        self.x_center = center_iv.begin
        self.s_center = set()
        s_left = []
        s_right = []
        for k in intervals:
            if k.end <= self.x_center:
                s_left.append(k)
            elif k.begin > self.x_center:
                s_right.append(k)
            else:
                self.s_center.add(k)
        self.left_node = Node.from_sorted_intervals(s_left)
        self.right_node = Node.from_sorted_intervals(s_right)
        return self.rotate()

    @cython.cfunc
    def center_hit(self, interval):
        return interval.contains_point(self.x_center)

    @cython.cfunc
    def hit_branch(self, interval):
        return interval.begin > self.x_center

    @cython.cfunc
    def refresh_balance(self):
        left_depth = self.left_node.depth if self.left_node else 0
        right_depth = self.right_node.depth if self.right_node else 0
        self.depth = 1 + max(left_depth, right_depth)
        self.balance = right_depth - left_depth

    def compute_depth(self):
        left_depth = self.left_node.compute_depth() if self.left_node else 0
        right_depth = self.right_node.compute_depth() if self.right_node else 0
        return 1 + max(left_depth, right_depth)

    @cython.cfunc
    def rotate(self):
        self.refresh_balance()
        if abs(self.balance) < 2:
            return self
        my_heavy = self.balance > 0
        child_heavy = self[my_heavy].balance > 0
        if my_heavy == child_heavy or self[my_heavy].balance == 0:
            return self.srotate()
        else:
            return self.drotate()

    @cython.cfunc
    def srotate(self):
        heavy = self.balance > 0
        light = not heavy
        save = self[heavy]
        self[heavy] = save[light]
        save[light] = self.rotate()
        promotees = [iv for iv in save[light].s_center if save.center_hit(iv)]
        if promotees:
            for iv in promotees:
                save[light] = save[light].remove(iv)
            save.s_center.update(promotees)
        save.refresh_balance()
        return save

    @cython.cfunc
    def drotate(self):
        my_heavy = self.balance > 0
        self[my_heavy] = self[my_heavy].srotate()
        self.refresh_balance()
        result = self.srotate()
        return result

    @cython.cfunc
    def add(self, interval):
        if self.center_hit(interval):
            self.s_center.add(interval)
            return self
        else:
            direction = self.hit_branch(interval)
            if not self[direction]:
                self[direction] = Node.from_interval(interval)
                self.refresh_balance()
                return self
            else:
                self[direction] = self[direction].add(interval)
                return self.rotate()

    @cython.ccall
    def remove(self, interval):
        done = []
        return self.remove_interval_helper(interval, done, should_raise_error=True)

    @cython.ccall
    def discard(self, interval):
        done = []
        return self.remove_interval_helper(interval, done, should_raise_error=False)

    def remove_interval_helper(self, interval, done, should_raise_error):
        if self.center_hit(interval):
            if not should_raise_error and interval not in self.s_center:
                done.append(1)
                return self
            try:
                self.s_center.remove(interval)
            except:
                self.print_structure()
                raise KeyError(interval)
            if self.s_center:
                done.append(1)
                return self
            return self.prune()
        else:
            direction = self.hit_branch(interval)
            if not self[direction]:
                if should_raise_error:
                    raise ValueError
                done.append(1)
                return self
            self[direction] = self[direction].remove_interval_helper(interval, done, should_raise_error)
            if not done:
                return self.rotate()
            return self

    @cython.ccall
    def search_overlap(self, point_list):
        result = set()
        for j in point_list:
            self.search_point(j, result)
        return result

    @cython.ccall
    def search_point(self, point, result):
        for k in self.s_center:
            if k.begin <= point < k.end:
                result.add(k)
        if point < self.x_center and self[0]:
            return self[0].search_point(point, result)
        elif point > self.x_center and self[1]:
            return self[1].search_point(point, result)
        return result

    @cython.cfunc
    def prune(self):
        if not self[0] or not self[1]:
            direction = not self[0]
            result = self[direction]
            return result
        else:
            heir, self[0] = self[0].pop_greatest_child()
            heir[0], heir[1] = self[0], self[1]
            heir.refresh_balance()
            heir = heir.rotate()
            return heir

    @cython.cfunc
    def pop_greatest_child(self):
        if not self.right_node:
            ivs = sorted(self.s_center, key=attrgetter('end', 'begin'))
            max_iv = ivs.pop()
            new_x_center = self.x_center
            while ivs:
                next_max_iv = ivs.pop()
                if next_max_iv.end == max_iv.end: continue
                new_x_center = max(new_x_center, next_max_iv.end)
            def get_new_s_center():
                for iv in self.s_center:
                    if iv.contains_point(new_x_center): yield iv
            child = Node(new_x_center, get_new_s_center())
            self.s_center -= child.s_center
            if self.s_center:
                return child, self
            else:
                return child, self[0]
        else:
            greatest_child, self[1] = self[1].pop_greatest_child()
            for iv in set(self.s_center):
                if iv.contains_point(greatest_child.x_center):
                    self.s_center.remove(iv)
                    greatest_child.add(iv)
            if self.s_center:
                self.refresh_balance()
                new_self = self.rotate()
                return greatest_child, new_self
            else:
                new_self = self.prune()
                return greatest_child, new_self

    @cython.ccall
    def contains_point(self, p):
        for iv in self.s_center:
            if iv.contains_point(p):
                return True
        branch = self[p > self.x_center]
        return branch and branch.contains_point(p)

    @cython.ccall
    def all_children(self):
        return self.all_children_helper(set())

    def all_children_helper(self, result):
        result.update(self.s_center)
        if self[0]:
            self[0].all_children_helper(result)
        if self[1]:
            self[1].all_children_helper(result)
        return result

    @cython.ccall
    def verify(self, parents=None):
        if parents is None:
            parents = set()
        assert(isinstance(self.s_center, set))
        bal = self.balance
        assert abs(bal) < 2
        self.refresh_balance()
        assert bal == self.balance
        assert self.s_center
        for iv in self.s_center:
            assert hasattr(iv, 'begin')
            assert hasattr(iv, 'end')
            assert iv.begin < iv.end
            assert iv.overlaps(self.x_center)
            for parent in sorted(parents):
                assert not iv.contains_point(parent)
        if self[0]:
            assert self[0].x_center < self.x_center
            self[0].verify(parents.union([self.x_center]))
        if self[1]:
            assert self[1].x_center > self.x_center
            self[1].verify(parents.union([self.x_center]))

    def __getitem__(self, index):
        if index:
            return self.right_node
        else:
            return self.left_node

    def __setitem__(self, key, value):
        if key:
            self.right_node = value
        else:
            self.left_node = value

    def __str__(self):
        return "Node<{0}, depth={1}, balance={2}>".format(
            self.x_center,
            self.depth,
            self.balance
        )

    @cython.ccall
    def count_nodes(self):
        count = 1
        if self.left_node:
            count += self.left_node.count_nodes()
        if self.right_node:
            count += self.right_node.count_nodes()
        return count

    @cython.ccall
    def depth_score(self, n, m):
        if n == 0:
            return 0.0
        dopt = 1 + int(floor(l2(m)))
        f = 1 / float(1 + n - dopt)
        return f * self.depth_score_helper(1, dopt)

    def depth_score_helper(self, d, dopt):
        di = d - dopt
        if di > 0:
            count = di * len(self.s_center)
        else:
            count = 0
        if self.right_node:
            count += self.right_node.depth_score_helper(d + 1, dopt)
        if self.left_node:
            count += self.left_node.depth_score_helper(d + 1, dopt)
        return count

    @cython.ccall
    def print_structure(self, indent=0, tostring=False):
        nl = '\n'
        sp = indent * '    '
        rlist = [str(self) + nl]
        if self.s_center:
            for iv in sorted(self.s_center):
                rlist.append(sp + ' ' + repr(iv) + nl)
        if self.left_node:
            rlist.append(sp + '<:  ')
            rlist.append(self.left_node.print_structure(indent + 1, True))
        if self.right_node:
            rlist.append(sp + '>:  ')
            rlist.append(self.right_node.print_structure(indent + 1, True))
        result = ''.join(rlist)
        if tostring:
            return result
        else:
            print(result)