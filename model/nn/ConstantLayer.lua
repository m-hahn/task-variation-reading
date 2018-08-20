require 'nn'

local ConstantLayer, _ = torch.class('nn.ConstantLayer','nn.Module')

function ConstantLayer:__init(outputTensorOrTable)
 --  self.gradInput = torch.Tensor(0)
   self.output = outputTensorOrTable
   self.gradInput = torch.CudaTensor()
end

 -- expects the input to be a tensor
function ConstantLayer:updateOutput(input)
    --print("1112")
    --print(input)
    self.gradInput:resizeAs(input)
    self.gradInput:zero()
    return self.output
end


function ConstantLayer:updateGradInput(input, gradOutput)
   --print(input)
   return self.gradInput
   -- returning nil would lead to problems with

--   return self.gradInput
end

function ConstantLayer:accGradParameters(input, gradOutput)
end



