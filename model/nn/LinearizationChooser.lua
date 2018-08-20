require 'nn'

-- TODO

local LinearizationChooser, Parent = torch.class('nn.LinearizationChooser', 'nn.Module')

function LinearizationChooser:__init(outputSize)
   Parent.__init(self)
   self.outputSize = outputSize
   self.output = torch.Tensor(outputSize)
   crash()
end

function LinearizationChooser:updateOutput(input)
   self.output = torch.randn(self.outputSize)
   return self.output
end

function LinearizationChooser:updateGradInput(input, gradOutput)
   return {}
end


