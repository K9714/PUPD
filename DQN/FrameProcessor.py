class FrameProcessor():
    def __init__(self, frame_skip):
        self.frame_skip = frame_skip
        self.frame_stack = []
        self.next_frame_stack = []
        self.frame_count = 0
        self.next_frame_count = 0

    def __len__(self):
        return len(self.frame_stack[0])

    def frame_pop(self, idx=0):
        self.frame_count -= 1
        return self.frame_stack.pop(idx)
    
    def next_frame_pop(self, idx=0):
        self.next_frame_count -= 1
        return self.next_frame_stack.pop(idx)

    def frame_append(self, frame):
        for i in range(self.frame_count % self.frame_skip, len(self.frame_stack), self.frame_skip):
            self.frame_stack[i].append(frame)
        self.frame_stack.append([frame])
        self.frame_count += 1

    def next_frame_append(self, next_frame):
        for i in range(self.next_frame_count % self.frame_skip, len(self.next_frame_stack), self.frame_skip):
            self.next_frame_stack[i].append(next_frame)
        self.next_frame_stack.append([next_frame])
        self.next_frame_count += 1