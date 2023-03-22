



from _util.util_v1 import * ; import _util.util_v1 as util
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d


class DatabackendDanbooruTar:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(dtypes=None),  # dtypes=None ==> all
    )
    def __init__(self, args=None, collate=False, dk_meta=None):
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.dn = f'{self.args.base.dn}/_data/danbooru'
        # self.bns = uutil.safe_bns(self.get_bns())
        self.dtypes = set(self.args.load.dtypes) if self.args.load.dtypes!=None else set((
            'image', 'meta',
        ))

        # load metadata if needed
        self.dk_meta = dk_meta
        if 'meta' in self.dtypes and self.dk_meta is None:
            self.dk_meta = DatabaseDanbooruMeta(args=self.args_user)

        # load index
        self.fn_tar_index = f'{self.dn}/cache/tar_index.csv'
        self.fns_tar = [
            f'{self.dn}/raw/tars/{10*i:04}-{10*i+9:04}.tar'
            for i in range(100)
        ]
        self.tar_index = {}
        assert os.path.isfile(self.fn_tar_index)  # self.preprocess_tar_index()
        for line in util.read(self.fn_tar_index).split('\n'):
            q = line.split(',')
            self.tar_index[int(q[0])] = int(q[1]), int(q[2])  # id: start, size
        self.bns = uutil.safe_bns(sorted(self.tar_index.keys()))
        return
    def preprocess_tar_index(self):
        idx = []
        for i,tfn in enumerate(tqdm(self.fns_tar)):
            idx_sub = []
            with tarfile.open(tfn, 'r') as handle:
                for ti in handle:
                    if ti.isfile():
                        idx_sub.append( ','.join([
                            ti.name.split('/')[-1].split('.')[0], # id
                            str(ti.offset_data),  # start
                            str(ti.size),  # size
                        ]) )
            idx_sub = '\n'.join(idx_sub)
            util.write(idx_sub, f'{self.fn_tar_index}.{i:02d}.csv')
            idx.append(idx_sub)
        util.write('\n'.join(idx), self.fn_tar_index)
        return
    
    def __getitem__(self, bn, collate=None, return_more=False):
        bn = uutil.unsafe_bn(bn, bns=self.bns)
        ret = Dict({
            'bn': bn,
        })

        if 'image' in self.dtypes:
            bint = int(bn)
            ti = (bint%1000)//10
            start,size = self.tar_index[bint]
            try:
                with open(self.fns_tar[ti], 'rb') as handle:
                    handle.seek(start)
                    ans = handle.read(size)
                ret['image'] = I(Image.open(io.BytesIO(ans)).convert('RGBA'))
            except:
                ret['image'] = None

        if 'meta' in self.dtypes:
            ret['meta'] = self.dk_meta[bn]
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret
    
    def __len__(self):
        return len(self.bns)

    def keys(self):
        return self.bns

class DatabackendDanbooruWeb:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(dtypes=None, retry=(10,20)),  # dtypes=None ==> all
    )
    def __init__(self, args=None, collate=False, use_key=False):
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.use_key = use_key
        if self.use_key:
            assert 'DANBOORU_API_KEY' in os.environ and 'DANBOORU_API_USER' in os.environ
        self.dn = f'{self.args.base.dn}/_data/danbooru'
        self.dtypes = set(self.args.load.dtypes) if self.args.load.dtypes!=None else set((
            'image', 'meta', 'status_code',
        ))
        return
    def _danbooru_get(self, bn, meta=None):
        # get meta
        if meta is None:
            x = requests.get(f'https://danbooru.donmai.us/posts/{bn}.json')
            status_meta = x.status_code
            if x.status_code==429:  # throttled
                a,b = self.args.load.retry
                time.sleep(a+np.random.rand()*b)
                return self._danbooru_get(bn)
            meta = x.json()
        
        # get image
        x = requests.get(meta['file_url'])
        status_image = x.status_code
        if x.status_code==429:  # throttled
            a,b = self.args.load.retry
            time.sleep(a+np.random.rand()*b)
            return self._danbooru_get(bn, meta=meta)
        return Dict({
            # 'bn': str(bn),
            'image': I(Image.open(io.BytesIO(x.content))),
            'meta': meta,
            'status_code': {
                'image': status_image,
                'meta': status_meta,
            },
        })
    def counts(self, tags, use_key=None):
        use_key = self.use_key if use_key is None else use_key
        qargs = [
            # 'page=1',
            # 'limit=200',
            *([] if len(tags)==0 else [
                'tags='+'+'.join([
                    urllib.parse.quote_plus(tag)
                    for tag in tags
                ]),
            ]),
            *([] if not use_key else [
                f'api_key={os.environ["DANBOORU_API_KEY"]}',
                f'login={os.environ["DANBOORU_API_USER"]}',
            ]),
        ]
        x = requests.get('https://danbooru.donmai.us/counts/posts.json?'+'&'.join(qargs))
        if x.status_code==429:  # throttled
            a,b = self.args.load.retry
            time.sleep(a+np.random.rand()*b)
            return self.counts(tags, use_key=use_key)
        x = x.json()['counts']['posts']
        return x
    def search(self, tags, use_key=None):
        use_key = self.use_key if use_key is None else use_key
        page = 1
        qargs = [
            # 'page=1',
            # 'limit=200',
            *([] if len(tags)==0 else [
                'tags='+'+'.join([
                    urllib.parse.quote_plus(tag)
                    for tag in tags
                ]),
            ]),
            *([] if not use_key else [
                f'api_key={os.environ["DANBOORU_API_KEY"]}',
                f'login={os.environ["DANBOORU_API_USER"]}',
            ]),
        ]
        seen = set()
        while True:
            x = requests.get(
                'https://danbooru.donmai.us/posts.json?' +
                '&'.join([f'page={page}']+qargs)
            )
            if x.status_code==429:  # throttled
                a,b = self.args.load.retry
                time.sleep(a+np.random.rand()*b)
                continue
            x = x.json()
            if len(x)==0:
                break
            else:
                for i in range(len(x)):
                    if x[i]['id'] not in seen:
                        yield x[i]
                        seen.add(x[i]['id'])
                    else:
                        continue
                page += 1
        return

    def __getitem__(self, bn, collate=None, return_more=False):
        bn = int(bn)
        ret = Dict({
            'bn': str(bn),
        })
        
        if 'image' in self.dtypes or 'meta' in self.dtypes or 'status_code' in self.dtypes:
            gotten = self._danbooru_get(bn)

        if 'image' in self.dtypes:
            ret['image'] = gotten['image']
        if 'meta' in self.dtypes:
            ret['meta'] = gotten['meta']
        if 'status_code' in self.dtypes:
            ret['status_code'] = gotten['status_code']
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret
    def __len__(self):
        x = requests.get(f'https://danbooru.donmai.us/counts/posts.json')
        if x.status_code==429:  # throttled
            a,b = self.args.load.retry
            time.sleep(a+np.random.rand()*b)
            return self.__len__()
        return x.json()['counts']['posts']

class DatabaseDanbooruMeta:
    db_attributes = [
        'id',
        'pixiv_id',

        'file_ext',
        'image_width',
        'image_height',
        'file_size',
        'md5',

        'source',
        'parent_id',
        'has_children',

        'uploader_id',
        'approver_id',
        'rating',
        'score',
        'up_score',
        'down_score',
        'favs',

        'created_at',
        'updated_at',
        'last_commented_at',
        'last_noted_at',

        'is_note_locked',
        'is_rating_locked',
        'is_status_locked',
        'is_pending',
        'is_flagged',
        'is_deleted',
        'is_banned',
    ]
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        # load=Dict(dtypes=None),  # dtypes=None ==> all
    )
    def __init__(self, args=None):
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.fn_db = f'{self.args.base.dn}/_data/danbooru/cache/metadata.db'
        self.dn_raw = f'{self.args.base.dn}/_data/danbooru/raw/metadata/2019'
        self.fns_raw = sorted([
            f'{self.dn_raw}/{fn}'
            for fn in os.listdir(self.dn_raw)
        ]) if os.path.isdir(self.dn_raw) else []
        self.uri_db = f'file:{self.fn_db}?mode=ro'
        return
    def tag(self, query):
        try:
            query = int(query)
            byid = True
        except:
            byid = False
        with sqlite3.connect(self.uri_db, uri=True) as conn:
            c = conn.cursor()
            if byid:
                ans = self['select name from tag where id=:query', {'query': query}]
            else:
                ans = self['select id from tag where name=:query', {'query': query}]
        return ans[0][0]
    def category(self, query):
        try:
            query = int(query)
            byid = True
        except:
            byid = False
        with sqlite3.connect(self.uri_db, uri=True) as conn:
            c = conn.cursor()
            if byid:
                ans = self['select category from tag where id=:query', {'query': query}]
            else:
                ans = self['select category from tag where name=:query', {'query': query}]
        return ans[0][0]

    def __getitem__(self, idx):
        try:
            idx = int(idx)
            query = False
        except:
            query = True
        if not query:
            ans = {
                'url': f'https://danbooru.donmai.us/posts/{idx}'
            }
            with sqlite3.connect(self.uri_db, uri=True) as conn:
                c = conn.cursor()

                # regular metadata
                c.execute(f"""
                SELECT {','.join(self.db_attributes)} FROM image WHERE id=:idx;
                """, {'idx': idx})
                for k,q in zip(self.db_attributes,c.fetchall()[0]):
                    ans[k] = q
                    
                # tags
                c.execute(f"""
                SELECT id_tag FROM image_tag WHERE id_image=:idx;
                """, {'idx': idx})
                ans['tags'] = [q[0] for q in c.fetchall()]
                
                # pools
                c.execute(f"""
                SELECT id_pool FROM image_pool WHERE id_image=:idx;
                """, {'idx': idx})
                ans['pools'] = [q[0] for q in c.fetchall()]
            return Dict(ans)
        else:
            ans = []
            with sqlite3.connect(self.uri_db, uri=True) as conn:
                c = conn.cursor()
                if not isinstance(idx, list):
                    idx = [idx,]
                for q in idx:
                    if isinstance(q, tuple):
                        c.execute(q[0], q[1])
                    else:
                        c.execute(q)
                    ans.append(c.fetchall())
            if len(idx)==1:
                ans = ans[0]
            return ans


    # UNTESTED
    def preprocess(self, force=False, dn_temp='/dev/shm'):
        if not force and os.path.isfile(self.fn_db): return

        # create table
        fn_temp = f'{dn_temp}/metadata.db'
        with sqlite3.connect(fn_temp) as conn:
            c = conn.cursor()
            c.execute("""DROP TABLE IF EXISTS image;""")
            c.execute("""DROP TABLE IF EXISTS tag;""")
            c.execute("""DROP TABLE IF EXISTS image_tag;""")
            c.execute("""DROP TABLE IF EXISTS image_pool;""")
            c.execute("""CREATE TABLE IF NOT EXISTS image (
                id int PRIMARY KEY,
                pixiv_id int,
                
                file_ext text,
                image_width int,
                image_height int,
                file_size int,
                md5 text,
                
                source text,
                parent_id int,
                has_children int,
                
                uploader_id int,
                approver_id int,
                rating text,
                score int,
                up_score int,
                down_score int,
                favs int,
                
                created_at datetime,
                updated_at datetime,
                last_commented_at datetime,
                last_noted_at datetime,
                
                is_note_locked int,
                is_rating_locked int,
                is_status_locked int,
                is_pending int,
                is_flagged int,
                is_deleted int,
                is_banned int
            );""")
            c.execute("""CREATE TABLE IF NOT EXISTS tag (
                id int PRIMARY KEY,
                name text,
                category int
            );""")
            c.execute("""CREATE TABLE IF NOT EXISTS image_tag (
                id_image int,
                id_tag int,
                PRIMARY KEY (id_image, id_tag)
            );""")
            c.execute("""CREATE TABLE IF NOT EXISTS image_pool (
                id_image int,
                id_pool int,
                PRIMARY KEY (id_image, id_pool)
            );""")

        # perform db ops
        cmds = []
        def _doit(cmds):
            with sqlite3.connect(fn_temp) as conn:
                c = conn.cursor()
                for cmd,params in cmds:
                    c.execute(cmd, params)
            return
        for fn_meta in tqdm(self.fns_raw):
            with open(fn_meta, 'r') as handle:
                metas = [
                    json.loads(line)
                    for line in handle
                ]
            for m in metas:
                cmds.append((f"""
                INSERT OR IGNORE INTO image (
                    id,
                    pixiv_id,

                    file_ext,
                    image_width,
                    image_height,
                    file_size,
                    md5,

                    source,
                    parent_id,
                    has_children,

                    uploader_id,
                    approver_id,
                    rating,
                    score,
                    up_score,
                    down_score,
                    favs,

                    created_at,
                    updated_at,
                    last_commented_at,
                    last_noted_at,

                    is_note_locked,
                    is_rating_locked,
                    is_status_locked,
                    is_pending,
                    is_flagged,
                    is_deleted,
                    is_banned
                ) VALUES (
                    {m['id']},
                    {m['pixiv_id']},

                    {repr(m['file_ext'])},
                    {m['image_width']},
                    {m['image_height']},
                    {m['file_size']},
                    {repr(m['md5'])},

                    :source,
                    {m['parent_id']},
                    {int(m['has_children'])},

                    {m['uploader_id']},
                    {m['approver_id']},
                    {repr(m['rating'])},
                    {m['score']},
                    {m['up_score']},
                    {m['down_score']},
                    {len(m['favs'])},

                    {repr(m['created_at'])},
                    {repr(m['updated_at'])},
                    {repr(m['last_commented_at'])},
                    {repr(m['last_noted_at'])},

                    {int(m['is_note_locked'])},
                    {int(m['is_rating_locked'])},
                    {int(m['is_status_locked'])},
                    {int(m['is_pending'])},
                    {int(m['is_flagged'])},
                    {int(m['is_deleted'])},
                    {int(m['is_banned'])}
                );""", {'source': m['source']}))
                for t in m['tags']:
                    cmds.append((f"""
                    INSERT OR IGNORE INTO tag (
                        id, name, category
                    ) VALUES (
                        {t['id']}, :name, {t['category']}
                    );""", {'name': t['name']}))
                    cmds.append((f"""
                    INSERT OR IGNORE INTO image_tag (
                        id_image, id_tag
                    ) VALUES (
                        {m['id']}, {t['id']}
                    );""", {}))
                for p in m['pools']:
                    cmds.append((f"""
                    INSERT OR IGNORE INTO image_pool (
                        id_image, id_pool
                    ) VALUES (
                        {m['id']}, :pool
                    );""", {'pool': p}))
                if len(cmds) > 10000000: # woah
                    _doit(cmds)
                    cmds = []
            _doit(cmds)
            cmds = []

        # copy completed from temp to real location
        shutil.copy(fn_temp, self.fn_db)
        return



