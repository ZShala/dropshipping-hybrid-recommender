import React from 'react';
import CategoryProducts from '../../components/category-products';
import SEO from '../../components/seo';

const Fragrance = () => {
    return (
        <>
            <SEO 
                title="Fragrances | Luxury Perfumes Collection"
                description="Explore our curated collection of luxury perfumes and fragrances. Shop designer scents with authentic product guarantee and Express shipping."
            />
            <div>
                <CategoryProducts categoryType="fragrance" />
            </div>
        </>
    );
};

export default Fragrance;
