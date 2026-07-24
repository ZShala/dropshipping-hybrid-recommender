import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FaShoppingCart } from 'react-icons/fa';
import './cart-icon.styles.scss';

const CartIcon = () => {
    const [itemCount, setItemCount] = useState(0);

    useEffect(() => {
        // Function to update the product count
        const updateCartCount = () => {
            try {
                const cart = JSON.parse(localStorage.getItem('cart') || '[]');
                const count = cart.reduce((total, item) => total + (item.quantity || 0), 0);
                setItemCount(count);
            } catch (error) {
                console.error('Error updating cart count:', error);
                setItemCount(0);
            }
        };

        // Update the initial count
        updateCartCount();

        // Listen for changes in localStorage
        const handleStorageChange = (e) => {
            if (e.key === 'cart') {
                updateCartCount();
            }
        };

        window.addEventListener('storage', handleStorageChange);
        
        // Listen for a custom event for local updates
        window.addEventListener('cartUpdated', updateCartCount);

        return () => {
            window.removeEventListener('storage', handleStorageChange);
            window.removeEventListener('cartUpdated', updateCartCount);
        };
    }, []);

    return (
        <Link to="/cart" className="cart-icon-container">
            <FaShoppingCart className="shopping-icon" />
            {itemCount > 0 && <span className="item-count">{itemCount}</span>}
        </Link>
    );
};

export default CartIcon; 