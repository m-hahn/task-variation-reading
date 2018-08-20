-- from Jianpeng's code
-- https://github.com/cheng6076/SNLI-attention/blob/master/util/ReplicateAdd.lua

--[1,0,2,3  +  [1,2  =  [2,2,3,5
-- 2,0,1,3]    [2,3]     4,3,3,6]

local ReplicateAdd, parent = torch.class('nn.ReplicateAdd', 'nn.Module')

function ReplicateAdd:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

-- assumes that the outermost dimension is batch dimension
function ReplicateAdd:updateOutput(input)
    local vectors, scalars = unpack(input)
-- presumably more efficient by repeating into "self.output" and then add "vectors"

    self.output:repeatTensor(scalars, 1, vectors:size(2)/scalars:size(2)):add(vectors)
-------------------------
-- Jianpeng's code:
--    self.output:resizeAs(vectors):copy(vectors):add(torch.repeatTensor(scalars, 1, vectors:size(2)/scalars:size(2)))
------------------------
    return self.output
end

-- use of squeeze is dangerous if there are nontrivial/"coincidental" singleton dimensions
function ReplicateAdd:updateGradInput(input, gradOutput)
    local vectors, scalars = unpack(input)
    self.gradInput[1]:set(gradOutput)

    -- presumably more efficient by doing fully in place
    self.gradInput[2]:resizeAs(gradOutput):copy(gradOutput)
    self.gradInput[2] = self.gradInput[2]:view(scalars:size(1), -1, scalars:size(2)):sum(2):squeeze()

----------------------------
    -- Jianpeng's code
 --   local tmp = gradOutput:clone():view(scalars:size(1), -1, scalars:size(2)):sum(2):squeeze()
--    self.gradInput[2]:set(tmp)
-----------------------------

    return self.gradInput
end





