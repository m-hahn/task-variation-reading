-- from Jianpeng's code
-- https://github.com/cheng6076/SNLI-attention/blob/master/util/ReplicateDynamicAdd.lua

--[1,0,2,3  Rep  [1,2  =  [1,2,1,2
-- 2,0,1,3]    [2,3]     2,3,2,3]

local ReplicateDynamic, parent = torch.class('nn.ReplicateDynamic', 'nn.Module')

function ReplicateDynamic:__init()
    parent.__init(self)
    self.gradInput = {torch.CudaTensor(), torch.CudaTensor()}
end

-- assumes that the outermost dimension is batch dimension
-- assumes vectors : (batch,sequence,otherstuff1)
--         scalars : (batch,otherstuff2)
function ReplicateDynamic:updateOutput(input)
    local vectors, scalars = unpack(input)
   assert(vectors:size(1) == scalars:size(1))
-- presumably more efficient by repeating into "self.output" and then add "vectors"
    self.output:repeatTensor(scalars, 1, vectors:size(2))
-------------------------
-- Jianpeng's code:
--    self.output:resizeAs(vectors):copy(vectors):add(torch.repeatTensor(scalars, 1, vectors:size(2)/scalars:size(2)))
------------------------
    return self.output
end

-- use of squeeze is dangerous if there are nontrivial/"coincidental" singleton dimensions
function ReplicateDynamic:updateGradInput(input, gradOutput)
    local vectors, scalars = unpack(input)
    self.gradInput[1]:resizeAs(input[1]):zero()

    -- presumably more efficient by doing a little more in place
    self.gradInput[2]:resizeAs(gradOutput):copy(gradOutput)
    self.gradInput[2] = self.gradInput[2]:view(scalars:size(1), -1, scalars:size(2)):sum(2):squeeze()

----------------------------
    -- Jianpeng's code
 --   local tmp = gradOutput:clone():view(scalars:size(1), -1, scalars:size(2)):sum(2):squeeze()
--    self.gradInput[2]:set(tmp)
-----------------------------

    return self.gradInput
end





