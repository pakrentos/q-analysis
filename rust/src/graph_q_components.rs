use std::collections::{BTreeSet, HashMap};
use hashbrown::HashSet;
use crate::types::{CliqueId, Clique, VertexId};


struct CliqueManager {
    all_cliques_by_size: HashMap<usize, Vec<CliqueId>>, // size -> list of clique IDs
    clique_to_id: HashMap<Clique, CliqueId>,
    id_to_clique: Vec<Clique>, // CliqueId is an index into this Vec
}

impl CliqueManager {
    fn new() -> Self {
        CliqueManager {
            all_cliques_by_size: HashMap::new(),
            clique_to_id: HashMap::new(),
            id_to_clique: Vec::new(),
        }
    }

    // Returns (CliqueId, is_new)
    fn add_clique(&mut self, clique: Clique) -> (CliqueId, bool) {
        if let Some(&id) = self.clique_to_id.get(&clique) {
            return (id, false);
        }
        let clique_id = self.id_to_clique.len();
        let clique_size = clique.len();

        self.clique_to_id.insert(clique.clone(), clique_id);
        self.id_to_clique.push(clique.clone());
        self.all_cliques_by_size
            .entry(clique_size)
            .or_default()
            .push(clique_id);
        (clique_id, true)
    }

    fn get_clique_by_id(&self, clique_id: CliqueId) -> Option<&Clique> {
        self.id_to_clique.get(clique_id)
    }

    fn get_id_by_clique(&self, clique: &Clique) -> Option<&CliqueId> {
        self.clique_to_id.get(clique)
    }
    
    fn get_cliques_of_size(&self, size: usize) -> Option<&Vec<CliqueId>> {
        self.all_cliques_by_size.get(&size)
    }
}

struct Dsu {
    parent: HashMap<CliqueId, CliqueId>,
    // For root nodes:
    vertex_union_data: HashMap<CliqueId, BTreeSet<VertexId>>, // Root CliqueId -> Union of vertices
    clique_ids_in_component: HashMap<CliqueId, HashSet<CliqueId>>, // Root CliqueId -> Set of CliqueIds in this component
    size: HashMap<CliqueId, usize>, // For union by size heuristic
}

impl Dsu {
    fn new() -> Self {
        Dsu {
            parent: HashMap::new(),
            vertex_union_data: HashMap::new(),
            clique_ids_in_component: HashMap::new(),
            size: HashMap::new(),
        }
    }

    fn add_set(&mut self, clique_id: CliqueId, clique_data: &Clique) {
        if !self.parent.contains_key(&clique_id) {
            self.parent.insert(clique_id, clique_id);
            self.vertex_union_data.insert(clique_id, clique_data.clone());
            let mut initial_clique_ids = HashSet::new();
            initial_clique_ids.insert(clique_id);
            self.clique_ids_in_component.insert(clique_id, initial_clique_ids);
            self.size.insert(clique_id, 1);
        }
    }

    fn find(&mut self, clique_id: CliqueId) -> Option<CliqueId> {
        // Check if clique_id is known before trying to access parent map
        if !self.parent.contains_key(&clique_id) {
            return None;
        }
        let p = self.parent[&clique_id]; // Safe due to check above
        if p == clique_id {
            return Some(clique_id);
        }
        let root = self.find(p)?;
        self.parent.insert(clique_id, root);
        Some(root)
    }

    fn union(&mut self, id1: CliqueId, id2: CliqueId) {
        let root1_opt = self.find(id1);
        let root2_opt = self.find(id2);

        if root1_opt.is_none() || root2_opt.is_none() { return; }
        let mut root1 = root1_opt.unwrap();
        let mut root2 = root2_opt.unwrap();

        if root1 != root2 {
            if self.size.get(&root1).unwrap_or(&0) < self.size.get(&root2).unwrap_or(&0) {
                std::mem::swap(&mut root1, &mut root2);
            }
            
            self.parent.insert(root2, root1);
            
            if let Some(root2_vertex_data) = self.vertex_union_data.remove(&root2) {
                let root1_vertex_data = self.vertex_union_data.entry(root1).or_default();
                root1_vertex_data.extend(root2_vertex_data);
            }

            if let Some(root2_clique_ids) = self.clique_ids_in_component.remove(&root2) {
                let root1_clique_ids = self.clique_ids_in_component.entry(root1).or_default();
                root1_clique_ids.extend(root2_clique_ids);
            }
            
            let size2 = self.size.remove(&root2).unwrap_or(0);
            *self.size.entry(root1).or_insert(1) += size2;
        }
    }
    
    fn get_all_components_vertex_union_data(&self) -> Vec<BTreeSet<VertexId>> {
        self.parent.keys()
            .filter_map(|&clique_id| {
                if self.parent.get(&clique_id) == Some(&clique_id) { // Is root
                    self.vertex_union_data.get(&clique_id).cloned()
                } else {
                    None
                }
            })
            .collect()
    }

    fn get_all_components_clique_ids(&self) -> Vec<HashSet<CliqueId>> {
         self.parent.keys()
            .filter_map(|&clique_id| {
                if self.parent.get(&clique_id) == Some(&clique_id) { // Is root
                    self.clique_ids_in_component.get(&clique_id).cloned()
                } else {
                    None
                }
            })
            .collect()
    }
}

fn combinations_recursive(
    elements: &[VertexId],
    k: usize,
    start_index: usize,
    current_combination: &mut Vec<VertexId>,
    result: &mut Vec<Clique>,
) {
    if current_combination.len() == k {
        result.push(current_combination.iter().cloned().collect::<Clique>());
        return;
    }
    if start_index >= elements.len() {
        return;
    }

    for i in start_index..elements.len() {
        if k - current_combination.len() > elements.len() - i {
            break;
        }
        current_combination.push(elements[i]);
        combinations_recursive(elements, k, i + 1, current_combination, result);
        current_combination.pop();
    }
}

fn get_subcliques(clique_obj: &Clique, target_size: usize) -> Vec<Clique> {
    if target_size == 0 || target_size > clique_obj.len() {
        return Vec::new();
    }
    if target_size == clique_obj.len() {
        return vec![clique_obj.clone()];
    }

    let elements: Vec<VertexId> = clique_obj.iter().cloned().collect();
    let mut result = Vec::new();
    let mut current_combination = Vec::with_capacity(target_size);
    
    combinations_recursive(&elements, target_size, 0, &mut current_combination, &mut result);
    result
}

fn enumerate_k_cliques_from_graph(
    adj_map: &HashMap<VertexId, BTreeSet<VertexId>>,
    nodes_list_orig: &[VertexId],
    max_k_to_find: usize,
) -> (CliqueManager, usize) {
    let mut clique_manager = CliqueManager::new();

    if max_k_to_find == 0 || nodes_list_orig.is_empty() {
        return (clique_manager, 0);
    }

    let mut nodes_list = nodes_list_orig.to_vec();
    nodes_list.sort_unstable(); 

    for &v_node in &nodes_list {
        let mut clique = BTreeSet::new();
        clique.insert(v_node);
        clique_manager.add_clique(clique);
    }
    
    if max_k_to_find == 1 {
        let k_max_found = if clique_manager.get_cliques_of_size(1).map_or(false, |v| !v.is_empty()) { 1 } else { 0 };
        return (clique_manager, k_max_found);
    }

    for k_current in 1..max_k_to_find {
        let cliques_k_ids = match clique_manager.get_cliques_of_size(k_current) {
            Some(ids) => ids.clone(), 
            None => break, 
        };

        if cliques_k_ids.is_empty() {
            continue;
        }

        for &clique_k_id in &cliques_k_ids {
            let clique_k_obj = match clique_manager.get_clique_by_id(clique_k_id) {
                 Some(c) => c.clone(), 
                 None => continue, 
            };

            if clique_k_obj.is_empty() { continue; }

            let v_max_in_clique = clique_k_obj.iter().max().cloned().unwrap_or(0);
            let v_first = *clique_k_obj.iter().next().unwrap();
            
            let mut common_neighbors_candidates: BTreeSet<VertexId> = adj_map
                .get(&v_first)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|&neighbor| neighbor > v_max_in_clique)
                .collect();

            if common_neighbors_candidates.is_empty() { continue; }

            for &v_member in clique_k_obj.iter() {
                if v_member == v_first { continue; }
                
                if let Some(neighbors_of_v_member) = adj_map.get(&v_member) {
                    common_neighbors_candidates.retain(|cand| neighbors_of_v_member.contains(cand));
                } else {
                    common_neighbors_candidates.clear();
                }
                if common_neighbors_candidates.is_empty() { break; }
            }
            
            for &neighbor_node in &common_neighbors_candidates {
                let mut new_clique_members = clique_k_obj.clone();
                new_clique_members.insert(neighbor_node);
                clique_manager.add_clique(new_clique_members); // Adds (k_current+1)-clique
            }
        }
    }
    
    let mut k_max_found = 0;
    for k_val in (1..=max_k_to_find).rev() {
        if let Some(cliques) = clique_manager.get_cliques_of_size(k_val) {
            if !cliques.is_empty() {
                k_max_found = k_val;
                break;
            }
        }
    }
    (clique_manager, k_max_found)
}

#[allow(clippy::too_many_lines)]
pub fn find_all_q_connected_components(
    graph_adj_map: &HashMap<VertexId, BTreeSet<VertexId>>,
    k_max_hint_param: Option<usize>,
) -> HashMap<isize, Vec<BTreeSet<VertexId>>> {
    let graph_nodes = graph_adj_map.keys().cloned().collect::<Vec<VertexId>>();
    let k_max_hint = k_max_hint_param.unwrap_or_else(|| graph_nodes.len());
    
    let effective_k_max_hint = if k_max_hint == 0 && !graph_nodes.is_empty() { 1 } else { k_max_hint };
    
    let (clique_manager, k_max_actual) =
        enumerate_k_cliques_from_graph(graph_adj_map, &graph_nodes, effective_k_max_hint);

    let mut results_by_q: HashMap<isize, Vec<BTreeSet<VertexId>>> = HashMap::new();
    
    if k_max_actual == 0 {
        // If graph_nodes is empty, k_max_actual will be 0. Loop for q will be empty. Correct.
        // If graph_nodes is not empty, but k_max_actual is 0 (e.g. k_max_hint was 0 initially, and effective_k_max_hint also ended up 0, or no cliques at all),
        // q_max will be -1. Loop for q will be empty. Correct.
        // The Python code has a specific `if k_max_actual == 0 and len(graph_nodes) > 0: k_max_actual = 1`.
        // My `effective_k_max_hint` logic and `enumerate_k_cliques_from_graph` aims to make `k_max_actual >= 1`
        // if nodes exist and hint allows. If `k_max_actual` is still 0, it implies no 1-cliques could be formed or found.
        return results_by_q; 
    }

    let q_max = k_max_actual as isize - 1;
    let mut dsu_prev_q_level: Option<Dsu> = None;

    for q_isize in (0..=q_max).rev() {
        let q = q_isize as usize; 
        let kc = q + 1; // Clique size for q-connectivity

        let mut dsu_current_q = Dsu::new();

        let kc_clique_ids_option = clique_manager.get_cliques_of_size(kc);

        if kc_clique_ids_option.map_or(true, |ids| ids.is_empty()) {
            results_by_q.insert(q_isize, Vec::new());
            dsu_prev_q_level = Some(dsu_current_q); 
            continue;
        }
        
        let kc_clique_ids = kc_clique_ids_option.unwrap(); // Safe due to check above

        for &clique_id in kc_clique_ids {
            if let Some(clique_obj) = clique_manager.get_clique_by_id(clique_id) {
                dsu_current_q.add_set(clique_id, clique_obj);
            }
        }

        if let Some(ref dsu_prev) = dsu_prev_q_level {
            let components_prev_level_clique_ids = dsu_prev.get_all_components_clique_ids();
            for comp_kc_plus_1_clique_ids_set in components_prev_level_clique_ids {
                let mut first_subclique_id_for_this_comp: Option<CliqueId> = None;
                let mut all_subclique_ids_for_this_comp = Vec::new();

                for &k_prime_id in &comp_kc_plus_1_clique_ids_set { 
                    if let Some(k_prime_obj) = clique_manager.get_clique_by_id(k_prime_id) {
                        let kc_subcliques_of_k_prime = get_subcliques(k_prime_obj, kc);
                        
                        for k_sub_obj in kc_subcliques_of_k_prime { 
                            if let Some(&k_sub_id) = clique_manager.get_id_by_clique(&k_sub_obj) {
                                if dsu_current_q.parent.contains_key(&k_sub_id) {
                                     all_subclique_ids_for_this_comp.push(k_sub_id);
                                    if first_subclique_id_for_this_comp.is_none() {
                                        first_subclique_id_for_this_comp = Some(k_sub_id);
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(anchor_id) = first_subclique_id_for_this_comp {
                    for &other_id in &all_subclique_ids_for_this_comp {
                        if anchor_id != other_id { // Avoid self-union if it's the only one
                           dsu_current_q.union(anchor_id, other_id);
                        }
                    }
                }
            }
        }
        
        for k_l_size in (kc + 1)..=k_max_actual {
            if let Some(k_l_clique_ids) = clique_manager.get_cliques_of_size(k_l_size) {
                for &k_l_id in k_l_clique_ids {
                    if let Some(k_l_obj) = clique_manager.get_clique_by_id(k_l_id) { 
                        let kc_subcliques_of_k_l = get_subcliques(k_l_obj, kc); 

                        if kc_subcliques_of_k_l.len() > 1 {
                            let mut valid_kc_sub_ids_in_kl = Vec::new();
                            for sub_clique_obj in kc_subcliques_of_k_l {
                                if let Some(&sub_clique_id) = clique_manager.get_id_by_clique(&sub_clique_obj) {
                                    if dsu_current_q.parent.contains_key(&sub_clique_id) {
                                        valid_kc_sub_ids_in_kl.push(sub_clique_id);
                                    }
                                }
                            }

                            if valid_kc_sub_ids_in_kl.len() > 1 {
                                let first_kc_sub_id_in_kl = valid_kc_sub_ids_in_kl[0];
                                for i in 1..valid_kc_sub_ids_in_kl.len() {
                                    dsu_current_q.union(first_kc_sub_id_in_kl, valid_kc_sub_ids_in_kl[i]);
                                }
                            }
                        }
                    }
                }
            }
        }
        results_by_q.insert(q_isize, dsu_current_q.get_all_components_vertex_union_data());
        dsu_prev_q_level = Some(dsu_current_q);
    }
    results_by_q
}