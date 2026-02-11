def compute_loss(self, inputs, masks, epoch):
        # Forward Pass
        s_cls, s_patch, t_cls, t_patch = self.forward(inputs)
        
        # 1. DINO Loss (CLS) - Global Crops Only (Primele 2)
        loss_dino = (self.dino_loss(s_cls[0], t_cls[1], epoch) + 
                     self.dino_loss(s_cls[1], t_cls[0], epoch)) / 2
        
        # 2. iBOT Loss (Patch) - Global Crops Only
        # masks vine din DataLoader (o listă de măști pentru fiecare crop global)
        loss_ibot = 0
        if masks is not None:
             # Comparăm patch-urile Global 1 Student cu Global 1 Teacher (dar mascat)
             # Atenție: La iBOT comparăm aceleași vederi, studentul are mască, teacherul vede tot.
             l1 = self.ibot_loss(s_patch[0], t_patch[0], masks[0], epoch)
             l2 = self.ibot_loss(s_patch[1], t_patch[1], masks[1], epoch)
             loss_ibot = (l1 + l2) / 2

        # 3. Gram Loss - Încrucișat (Student 1 <-> Teacher 2)
        # Se aplică pe patch tokens NE-mascați (sau toți, dar DINOv3 preferă global structure)
        loss_gram = (self.gram_loss(s_patch[0], t_patch[1]) + 
                     self.gram_loss(s_patch[1], t_patch[0])) / 2
        
        # 4. KoLeo (Regularizare)
        loss_koleo = self.koleo_loss(s_cls[0])
        
        # Total Ponderat (Valori tipice DINOv3 config)
        # Gram Loss are de obicei un weight mic la început și crește, sau fix
        return loss_dino + 0.5 * loss_ibot + 0.1 * loss_koleo + 0.1 * loss_gram