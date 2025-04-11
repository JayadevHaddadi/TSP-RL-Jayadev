
    def augmentExamples(self, original_data):
        """
        For each (TSPState, pi, leftover_dist), produce (args.augmentationFactor - 1)
        additional variations using random label permutations and rotation around (0.5, 0.5).

        :param original_data: list of (state, pi, leftover_val).
        :return: the new extended list.
        """
        factor = getattr(self.args, "augmentationFactor", 1)
        if factor <= 1:
            return original_data

        augmented = []
        for state, pi, leftover in original_data:
            # Always keep the original
            augmented.append((state, pi, leftover))

            for _ in range(factor - 1):
                # 1) Apply a random permutation
                permuted_state, permuted_pi = self.applyRandomPermutation(state, pi)
                # 2) Apply rotation about (0.5, 0.5)
                rotated_state, rotated_pi = self.applyCenterRotation(
                    permuted_state, permuted_pi
                )
                # leftover remains the same if distance truly unchanged
                augmented.append((rotated_state, rotated_pi, leftover))

        return augmented
    
    def applyRandomPermutation(self, state, pi):
        """
        Randomly permute labels [0..n-1]. 
        We create new TSPState, reorder node coords accordingly,
        reorder the partial tour, and fix the unvisited array.
        Also reorder the pi distribution.
        """
        n = state.num_nodes
        perm = np.random.permutation(n)  # e.g. [2,0,1,...]
        
        old_coords = np.array(state.node_coordinates)
        new_coords = old_coords[perm].tolist()

        old_tour = state.tour
        new_tour = [perm[node] for node in old_tour]

        old_unvisited = state.unvisited
        new_unvisited = np.zeros_like(old_unvisited)
        for old_lbl in range(n):
            if old_unvisited[old_lbl] == 1:
                new_lbl = perm[old_lbl]
                new_unvisited[new_lbl] = 1

        from TSPState import TSPState

        # Create new distance matrix based on permutation
        if state.distance_matrix is not None:
            old_matrix = state.distance_matrix
            new_matrix = np.zeros_like(old_matrix)
            for i in range(n):
                for j in range(n):
                    new_matrix[perm[i]][perm[j]] = old_matrix[i][j]
        else:
            new_matrix = None

        new_state = TSPState(
            n,
            new_coords,
            distance_matrix=new_matrix,
            start_node=perm[state.tour[0]] if state.tour else None,
        )
        new_state.tour = new_tour
        new_state.unvisited = new_unvisited
        # if you recompute the partial cost, do:
        # new_state.current_length = self.recomputeTourLength(new_state)
        # otherwise copy:
        new_state.current_length = state.current_length

        # reorder pi
        new_pi = np.zeros(n, dtype=float)
        for old_label, prob in enumerate(pi):
            new_label = perm[old_label]
            new_pi[new_label] = prob
        
        return new_state, new_pi

    def applyCenterRotation(self, state, pi):
        """
        Rotate all coordinates about (0.5, 0.5) by a random angle in [0, 2*pi).
        Distances remain the same if TSP is in [0,1]^2. 
        We keep the same node labeling (tour/unvisited). 
        pi does not need label reorder, just the same array.

        If you want partial cost to remain identical, 
        either recalc or trust that the TSP code uses purely index-based cost 
        => same leftover is valid if everything in [0,1]^2 doesn't break distance.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        cosA, sinA = np.cos(angle), np.sin(angle)

        coords = state.node_coordinates
        rotated_coords = []
        for x, y in coords:
            # shift center to (0.0,0.0)
            dx = x - 0.5
            dy = y - 0.5
            # rotate
            rx = dx * cosA - dy * sinA
            ry = dx * sinA + dy * cosA
            # shift back
            rx += 0.5
            ry += 0.5
            rotated_coords.append([rx, ry])

        from TSPState import TSPState

        # For rotation, we need to recompute the distance matrix since distances change
        # But we can also keep the same distances if simplicity is preferred
        new_state = TSPState(
            state.num_nodes,
            rotated_coords,
            distance_matrix=state.distance_matrix,  # Reuse same matrix for simplicity
            start_node=state.tour[0] if state.tour else None,
        )
        new_state.tour = list(state.tour)
        new_state.unvisited = state.unvisited.copy()
        new_state.current_length = state.current_length

        return new_state, np.array(pi, copy=True)