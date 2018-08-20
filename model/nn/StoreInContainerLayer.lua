require 'nn'

recordMeans = true

local StoreInContainerLayer, _ = torch.class('nn.StoreInContainerLayer', 'nn.Module')

-- note does NOT try to infer minibatch size
function StoreInContainerLayer:__init(container,isString)
   if isString then
    self.containerName = container
   else
     self.result_container = container
   end
   assert(container ~= nil)

end


function StoreInContainerLayer:updateOutput(input)
   if self.containerName ~= nil then
     self.result_container = globalForExpOutput[self.containerName]
   end
   ---
   if recordMeans then
    if self.result_container.mean == nil then
      self.result_container.mean = 0
    end
    self.result_container.mean = 0.9999 * self.result_container.mean + 0.0001 * torch.mean(input)
   end
   ---
   self.output = input
   self.result_container.output = input


   return self.output
end

-- NOTE assumes input is a tensor, not a table
function StoreInContainerLayer:updateGradInput(input, gradOutput)
  if self.containerName ~= nil then
    self.result_container = globalForExpOutput[self.containerName]
  end
   self.gradInput = gradOutput
   self.result_container.grad = gradOutput
   return self.gradInput
end


function StoreInContainerLayer:clearState()
   self.result_container.output = nil
   self.result_container.grad = nil
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   return self
end

